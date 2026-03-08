import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { toast } from "sonner";
import { useAuth } from "@/context/AuthContext";
import {
  addAppStateListener,
  getCurrentTrackedPosition,
  getLocationRuntimeInfo,
  requestLocationPermissions,
  startTrackedLocationWatch,
} from "@/services/locationRuntime";
import { safetyApi } from "@/services/api";

const LOCATION_STORAGE_KEY = "alzahelp.location.current.v1";
const LOCATION_QUEUE_KEY = "alzahelp.location.queue.v1";
const MAX_QUEUE_SIZE = 12;
const MIN_PING_INTERVAL_MS = 30000;

const LocationTrackingContext = createContext(null);

const safeJsonParse = (value, fallback) => {
  try {
    return value ? JSON.parse(value) : fallback;
  } catch (_error) {
    return fallback;
  }
};

const loadStoredLocation = () => {
  if (typeof window === "undefined") return null;
  return safeJsonParse(window.localStorage.getItem(LOCATION_STORAGE_KEY), null);
};

const loadQueuedPings = () => {
  if (typeof window === "undefined") return [];
  const parsed = safeJsonParse(window.localStorage.getItem(LOCATION_QUEUE_KEY), []);
  return Array.isArray(parsed) ? parsed : [];
};

const getDocumentAppState = () => {
  if (typeof document === "undefined") return "foreground";
  return document.visibilityState === "hidden" ? "background" : "foreground";
};

const resolveErrorMessage = (error, fallback) => {
  if (!error) return fallback;
  if (typeof error === "string") return error;
  return error.message || fallback;
};

export const LocationTrackingProvider = ({ children }) => {
  const { user, isAuthenticated, loading } = useAuth();
  const location = useLocation();
  const runtimeInfo = useMemo(() => getLocationRuntimeInfo(), []);

  const [currentLocation, setCurrentLocation] = useState(() => loadStoredLocation());
  const [locating, setLocating] = useState(false);
  const [geoError, setGeoError] = useState("");
  const [permissionState, setPermissionState] = useState(runtimeInfo.native ? "native_managed" : "prompt");
  const [queueSize, setQueueSize] = useState(() => loadQueuedPings().length);
  const [lastSyncedAt, setLastSyncedAt] = useState(null);
  const [syncError, setSyncError] = useState("");
  const [isTracking, setIsTracking] = useState(false);

  const watcherCleanupRef = useRef(null);
  const appStateCleanupRef = useRef(null);
  const flushingRef = useRef(false);
  const queueRef = useRef(loadQueuedPings());
  const lastFlushAtRef = useRef(0);

  const persistLocation = useCallback((nextLocation) => {
    setCurrentLocation(nextLocation);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(LOCATION_STORAGE_KEY, JSON.stringify(nextLocation));
    }
  }, []);

  const persistQueue = useCallback((nextQueue) => {
    queueRef.current = nextQueue.slice(-MAX_QUEUE_SIZE);
    setQueueSize(queueRef.current.length);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(LOCATION_QUEUE_KEY, JSON.stringify(queueRef.current));
    }
  }, []);

  const flushQueuedPings = useCallback(
    async (force = false) => {
      if (flushingRef.current || queueRef.current.length === 0) return;
      if (typeof navigator !== "undefined" && navigator.onLine === false) return;
      if (!force && Date.now() - lastFlushAtRef.current < MIN_PING_INTERVAL_MS) return;

      flushingRef.current = true;
      try {
        while (queueRef.current.length > 0) {
          const payload = queueRef.current[0];
          const result = await safetyApi.pingLocation(payload);
          lastFlushAtRef.current = Date.now();
          setLastSyncedAt(new Date().toISOString());
          setSyncError("");
          if (result?.new_alerts?.length) {
            toast.error(`Safety alert: ${result.new_alerts[0].message}`);
          }
          persistQueue(queueRef.current.slice(1));
        }
      } catch (error) {
        setSyncError(resolveErrorMessage(error, "Could not sync location."));
      } finally {
        flushingRef.current = false;
      }
    },
    [persistQueue]
  );

  const enqueueLocationPing = useCallback(
    async (nextLocation, { force = false, source = "browser_watch", appState = "foreground" } = {}) => {
      if (nextLocation?.latitude == null || nextLocation?.longitude == null) return;
      const payload = {
        latitude: nextLocation.latitude,
        longitude: nextLocation.longitude,
        accuracy: nextLocation.accuracy ?? null,
        captured_at: nextLocation.updatedAt || new Date().toISOString(),
        source,
        app_state: appState,
      };
      persistQueue([...queueRef.current, payload]);
      await flushQueuedPings(force);
    },
    [flushQueuedPings, persistQueue]
  );

  const handlePositionUpdate = useCallback(
    async (position, meta = {}) => {
      const nextLocation = {
        latitude: position.latitude,
        longitude: position.longitude,
        accuracy: position.accuracy ?? null,
        updatedAt: position.updatedAt || meta.updatedAt || new Date().toISOString(),
      };
      persistLocation(nextLocation);
      setGeoError("");
      await enqueueLocationPing(nextLocation, {
        force: Boolean(meta.forceSync),
        source: meta.source || (runtimeInfo.native ? "native_track" : "browser_watch"),
        appState: meta.appState || getDocumentAppState(),
      });
      return nextLocation;
    },
    [enqueueLocationPing, persistLocation, runtimeInfo.native]
  );

  const stopTracking = useCallback(() => {
    const cleanup = watcherCleanupRef.current;
    watcherCleanupRef.current = null;
    if (typeof cleanup === "function") {
      Promise.resolve(cleanup()).catch(() => {});
    }
    setIsTracking(false);
  }, []);

  const stopAppStateListener = useCallback(() => {
    const cleanup = appStateCleanupRef.current;
    appStateCleanupRef.current = null;
    if (typeof cleanup === "function") {
      Promise.resolve(cleanup()).catch(() => {});
    }
  }, []);

  const refreshLocation = useCallback(async () => {
    setLocating(true);
    setGeoError("");
    try {
      const nextLocation = await getCurrentTrackedPosition();
      if (!nextLocation) {
        throw new Error("Could not get your location.");
      }
      return await handlePositionUpdate(nextLocation, {
        source: runtimeInfo.native ? "native_refresh" : "browser_refresh",
        forceSync: true,
        appState: "foreground",
      });
    } catch (error) {
      const message = resolveErrorMessage(error, "Could not get your location.");
      setGeoError(message);
      throw new Error(message);
    } finally {
      setLocating(false);
    }
  }, [handlePositionUpdate, runtimeInfo.native]);

  const shouldAutoTrack =
    !loading &&
    Boolean(isAuthenticated && user?.user_id && user?.role === "patient" && (runtimeInfo.native || location.pathname.startsWith("/dashboard")));

  useEffect(() => {
    if (runtimeInfo.native) {
      setPermissionState(runtimeInfo.hasBackgroundPlugin ? "native_background" : "native_managed");
      return undefined;
    }
    if (typeof navigator === "undefined" || !navigator.permissions?.query) return undefined;
    let cancelled = false;
    navigator.permissions
      .query({ name: "geolocation" })
      .then((status) => {
        if (cancelled) return;
        setPermissionState(status.state || "prompt");
        status.onchange = () => setPermissionState(status.state || "prompt");
      })
      .catch(() => {
        // Ignore unsupported permissions API.
      });
    return () => {
      cancelled = true;
    };
  }, [runtimeInfo.hasBackgroundPlugin, runtimeInfo.native]);

  useEffect(() => {
    stopTracking();
    if (!shouldAutoTrack) {
      return undefined;
    }

    let cancelled = false;
    const beginTracking = async () => {
      try {
        if (runtimeInfo.native) {
          await requestLocationPermissions().catch(() => null);
        }
        const cleanup = await startTrackedLocationWatch({
          watchOptions: {
            enableHighAccuracy: true,
            timeout: 15000,
            maximumAge: 15000,
            minimumUpdateInterval: MIN_PING_INTERVAL_MS,
          },
          onLocation: (nextLocation, meta = {}) => {
            if (cancelled) return;
            handlePositionUpdate(nextLocation, {
              source: meta.source || (runtimeInfo.native ? "native_track" : "browser_watch"),
              appState: meta.appState || getDocumentAppState(),
            }).catch(() => {
              // Passive tracking errors are surfaced via state.
            });
          },
          onError: (error) => {
            if (cancelled) return;
            setGeoError(resolveErrorMessage(error, "Could not watch your location."));
          },
        });
        if (cancelled) {
          Promise.resolve(cleanup?.()).catch(() => {});
          return;
        }
        watcherCleanupRef.current = cleanup;
        setIsTracking(true);
      } catch (error) {
        if (cancelled) return;
        setGeoError(resolveErrorMessage(error, "Could not watch your location."));
        setIsTracking(false);
      }
    };

    beginTracking();
    return () => {
      cancelled = true;
      stopTracking();
    };
  }, [handlePositionUpdate, runtimeInfo.native, shouldAutoTrack, stopTracking]);

  useEffect(() => {
    if (loading || !isAuthenticated || (!runtimeInfo.native && !location.pathname.startsWith("/dashboard"))) return undefined;
    let cancelled = false;
    safetyApi
      .getLocationState()
      .then((state) => {
        if (cancelled || !state?.latitude || !state?.longitude) return;
        if (!currentLocation || (currentLocation.updatedAt || "") < (state.captured_at || state.received_at || "")) {
          persistLocation({
            latitude: state.latitude,
            longitude: state.longitude,
            accuracy: state.accuracy ?? null,
            updatedAt: state.captured_at || state.received_at || new Date().toISOString(),
          });
          if (state.received_at) {
            setLastSyncedAt(state.received_at);
          }
        }
      })
      .catch(() => {
        // Best-effort hydration only.
      });
    return () => {
      cancelled = true;
    };
  }, [currentLocation, isAuthenticated, loading, location.pathname, persistLocation, runtimeInfo.native]);

  useEffect(() => {
    if (typeof window === "undefined" || typeof document === "undefined") return undefined;

    const handleOnline = () => {
      flushQueuedPings(true).catch(() => {});
    };
    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        flushQueuedPings(true).catch(() => {});
        if (shouldAutoTrack) {
          refreshLocation().catch(() => {});
        }
      } else if (currentLocation) {
        enqueueLocationPing(currentLocation, {
          force: true,
          source: "visibility_change",
          appState: "background",
        }).catch(() => {});
      }
    };

    window.addEventListener("online", handleOnline);
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      window.removeEventListener("online", handleOnline);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [currentLocation, enqueueLocationPing, flushQueuedPings, refreshLocation, shouldAutoTrack]);

  useEffect(() => {
    stopAppStateListener();
    let cancelled = false;
    const attach = async () => {
      const cleanup = await addAppStateListener((state) => {
        if (cancelled) return;
        if (state?.isActive) {
          flushQueuedPings(true).catch(() => {});
          if (shouldAutoTrack) {
            refreshLocation().catch(() => {});
          }
        } else if (currentLocation) {
          enqueueLocationPing(currentLocation, {
            force: true,
            source: state?.source || "native_app_state",
            appState: "background",
          }).catch(() => {});
        }
      });
      if (cancelled) {
        Promise.resolve(cleanup?.()).catch(() => {});
        return;
      }
      appStateCleanupRef.current = cleanup;
    };
    attach().catch(() => {});
    return () => {
      cancelled = true;
      stopAppStateListener();
    };
  }, [currentLocation, enqueueLocationPing, flushQueuedPings, refreshLocation, shouldAutoTrack, stopAppStateListener]);

  const value = useMemo(
    () => ({
      currentLocation,
      locating,
      geoError,
      permissionState,
      queueSize,
      lastSyncedAt,
      syncError,
      isTracking,
      runtimePlatform: runtimeInfo.platform,
      supportsNativeBackgroundTracking: runtimeInfo.hasBackgroundPlugin,
      trackingMode: shouldAutoTrack ? runtimeInfo.trackingMode : "manual_only",
      refreshLocation,
      flushQueuedPings: () => flushQueuedPings(true),
    }),
    [
      currentLocation,
      flushQueuedPings,
      geoError,
      isTracking,
      lastSyncedAt,
      locating,
      permissionState,
      queueSize,
      refreshLocation,
      runtimeInfo.hasBackgroundPlugin,
      runtimeInfo.platform,
      runtimeInfo.trackingMode,
      shouldAutoTrack,
      syncError,
    ]
  );

  return <LocationTrackingContext.Provider value={value}>{children}</LocationTrackingContext.Provider>;
};

export const useLocationTracking = () => {
  const context = useContext(LocationTrackingContext);
  if (!context) {
    throw new Error("useLocationTracking must be used within a LocationTrackingProvider");
  }
  return context;
};

export default LocationTrackingContext;
