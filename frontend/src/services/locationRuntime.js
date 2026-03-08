const DEFAULT_ONE_SHOT_OPTIONS = {
  enableHighAccuracy: true,
  timeout: 10000,
  maximumAge: 30000,
};

const DEFAULT_WATCH_OPTIONS = {
  enableHighAccuracy: true,
  timeout: 15000,
  maximumAge: 15000,
  minimumUpdateInterval: 30000,
};

const getWindow = () => (typeof window === "undefined" ? null : window);

const getCapacitor = () => getWindow()?.Capacitor || null;

const getCapacitorPlugin = (...names) => {
  const plugins = getCapacitor()?.Plugins || {};
  for (const name of names) {
    if (plugins?.[name]) {
      return plugins[name];
    }
  }
  return null;
};

const getNativePlatform = () => {
  const capacitor = getCapacitor();
  if (!capacitor) return "web";
  try {
    if (typeof capacitor.getPlatform === "function") {
      return capacitor.getPlatform() || "web";
    }
  } catch (_error) {
    return "web";
  }
  return capacitor.platform || "web";
};

const isFiniteNumber = (value) => Number.isFinite(Number(value));

const normalizeTimestamp = (value) => {
  if (value == null) return new Date().toISOString();
  const asDate = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(asDate.getTime())) {
    return new Date().toISOString();
  }
  return asDate.toISOString();
};

const normalizePosition = (rawPosition) => {
  if (!rawPosition) return null;
  const coords = rawPosition.coords || rawPosition;
  const latitude = coords?.latitude ?? coords?.lat;
  const longitude = coords?.longitude ?? coords?.lng ?? coords?.lon;
  if (!isFiniteNumber(latitude) || !isFiniteNumber(longitude)) {
    return null;
  }
  return {
    latitude: Number(latitude),
    longitude: Number(longitude),
    accuracy: isFiniteNumber(coords?.accuracy) ? Number(coords.accuracy) : null,
    updatedAt: normalizeTimestamp(rawPosition.timestamp || rawPosition.time || rawPosition.capturedAt),
  };
};

const toErrorMessage = (error, fallback) => {
  if (!error) return fallback;
  if (typeof error === "string") return error;
  return error.message || error.localizedMessage || fallback;
};

const isNativePlatform = () => {
  const capacitor = getCapacitor();
  if (!capacitor) return false;
  try {
    if (typeof capacitor.isNativePlatform === "function") {
      return Boolean(capacitor.isNativePlatform());
    }
  } catch (_error) {
    return false;
  }
  const platform = getNativePlatform();
  return platform === "ios" || platform === "android";
};

const getBackgroundPlugin = () =>
  getCapacitorPlugin("AlzaBackgroundLocation", "BackgroundGeolocation", "BackgroundLocation");

const getGeolocationPlugin = () => getCapacitorPlugin("Geolocation");

const getAppPlugin = () => getCapacitorPlugin("App");

const runBrowserGetCurrentPosition = (options = DEFAULT_ONE_SHOT_OPTIONS) =>
  new Promise((resolve, reject) => {
    if (typeof navigator === "undefined" || !navigator.geolocation) {
      reject(new Error("This device does not support location services."));
      return;
    }
    navigator.geolocation.getCurrentPosition(resolve, reject, options);
  });

const startBrowserWatch = ({ onLocation, onError, options = DEFAULT_WATCH_OPTIONS }) => {
  if (typeof navigator === "undefined" || !navigator.geolocation) {
    throw new Error("This device does not support location services.");
  }
  const watchId = navigator.geolocation.watchPosition(
    (position) => {
      const normalized = normalizePosition(position);
      if (normalized) {
        onLocation(normalized, { source: "browser_watch", appState: "foreground" });
      }
    },
    (error) => {
      onError(new Error(toErrorMessage(error, "Could not watch your location.")));
    },
    options
  );
  return async () => {
    navigator.geolocation.clearWatch(watchId);
  };
};

const startBackgroundPluginTracking = async (plugin, { onLocation, onError, watchOptions = DEFAULT_WATCH_OPTIONS }) => {
  if (typeof plugin.addWatcher === "function") {
    const watcherId = await plugin.addWatcher(
      {
        requestPermissions: true,
        stale: false,
        distanceFilter: 25,
        backgroundTitle: "AlzaHelp location monitoring",
        backgroundMessage: "Tracking location to keep safety alerts up to date.",
        ...watchOptions,
      },
      (location, error) => {
        if (error) {
          onError(new Error(toErrorMessage(error, "Native background tracking failed.")));
          return;
        }
        const normalized = normalizePosition(location);
        if (normalized) {
          onLocation(normalized, { source: "native_background", appState: "background" });
        }
      }
    );
    return async () => {
      if (typeof plugin.removeWatcher === "function") {
        try {
          await plugin.removeWatcher({ id: watcherId });
          return;
        } catch (_error) {
          await plugin.removeWatcher(watcherId);
        }
      }
    };
  }

  if (typeof plugin.addListener === "function" && (typeof plugin.startTracking === "function" || typeof plugin.start === "function")) {
    const listeners = [];
    const locationListener = await plugin.addListener("location", (location) => {
      const normalized = normalizePosition(location);
      if (normalized) {
        onLocation(normalized, { source: "native_background", appState: "background" });
      }
    });
    listeners.push(locationListener);

    if (typeof plugin.addListener === "function") {
      const errorListener = await plugin.addListener("error", (error) => {
        onError(new Error(toErrorMessage(error, "Native background tracking failed.")));
      });
      listeners.push(errorListener);
    }

    if (typeof plugin.startTracking === "function") {
      await plugin.startTracking({
        distanceFilter: 25,
        desiredAccuracy: "high",
        ...watchOptions,
      });
    } else {
      await plugin.start();
    }

    return async () => {
      if (typeof plugin.stopTracking === "function") {
        try {
          await plugin.stopTracking();
        } catch (_error) {
          // Best-effort cleanup.
        }
      } else if (typeof plugin.stop === "function") {
        try {
          await plugin.stop();
        } catch (_error) {
          // Best-effort cleanup.
        }
      }
      await Promise.all(
        listeners.map((listener) => {
          if (listener?.remove) {
            return listener.remove();
          }
          return Promise.resolve();
        })
      );
    };
  }

  throw new Error("No supported native background location plugin was found.");
};

const startNativeGeolocationWatch = async (plugin, { onLocation, onError, watchOptions = DEFAULT_WATCH_OPTIONS }) => {
  const watchId = await plugin.watchPosition(watchOptions, (position, error) => {
    if (error) {
      onError(new Error(toErrorMessage(error, "Native location watch failed.")));
      return;
    }
    const normalized = normalizePosition(position);
    if (normalized) {
      onLocation(normalized, { source: "native_geolocation", appState: "foreground" });
    }
  });
  return async () => {
    if (typeof plugin.clearWatch === "function") {
      await plugin.clearWatch({ id: watchId });
    }
  };
};

export const getLocationRuntimeInfo = () => {
  const native = isNativePlatform();
  const platform = getNativePlatform();
  const hasBackgroundPlugin = Boolean(getBackgroundPlugin());
  const hasNativeGeolocation = Boolean(getGeolocationPlugin());
  let trackingMode = "web_browser";
  if (native && hasBackgroundPlugin) {
    trackingMode = "native_background";
  } else if (native && hasNativeGeolocation) {
    trackingMode = "native_foreground";
  }
  return {
    native,
    platform,
    hasBackgroundPlugin,
    hasNativeGeolocation,
    trackingMode,
  };
};

export const requestLocationPermissions = async () => {
  const backgroundPlugin = getBackgroundPlugin();
  if (backgroundPlugin?.requestPermissions) {
    return backgroundPlugin.requestPermissions();
  }
  const geolocationPlugin = getGeolocationPlugin();
  if (geolocationPlugin?.requestPermissions) {
    return geolocationPlugin.requestPermissions();
  }
  return null;
};

export const getCurrentTrackedPosition = async (options = DEFAULT_ONE_SHOT_OPTIONS) => {
  const backgroundPlugin = getBackgroundPlugin();
  if (isNativePlatform() && backgroundPlugin) {
    if (typeof backgroundPlugin.getCurrentPosition === "function") {
      const position = await backgroundPlugin.getCurrentPosition(options);
      const normalized = normalizePosition(position);
      if (normalized) return normalized;
    }
    if (typeof backgroundPlugin.getCurrentLocation === "function") {
      const position = await backgroundPlugin.getCurrentLocation(options);
      const normalized = normalizePosition(position);
      if (normalized) return normalized;
    }
  }

  const geolocationPlugin = getGeolocationPlugin();
  if (isNativePlatform() && geolocationPlugin?.getCurrentPosition) {
    const position = await geolocationPlugin.getCurrentPosition(options);
    const normalized = normalizePosition(position);
    if (normalized) return normalized;
  }

  const browserPosition = await runBrowserGetCurrentPosition(options);
  return normalizePosition(browserPosition);
};

export const startTrackedLocationWatch = async ({
  onLocation,
  onError,
  oneShotOptions = DEFAULT_ONE_SHOT_OPTIONS,
  watchOptions = DEFAULT_WATCH_OPTIONS,
} = {}) => {
  const locationHandler = typeof onLocation === "function" ? onLocation : () => {};
  const errorHandler = typeof onError === "function" ? onError : () => {};
  const runtimeInfo = getLocationRuntimeInfo();

  try {
    const current = await getCurrentTrackedPosition(oneShotOptions);
    if (current) {
      locationHandler(current, {
        source: runtimeInfo.native ? (runtimeInfo.hasBackgroundPlugin ? "native_refresh" : "native_geolocation") : "browser_refresh",
        appState: "foreground",
      });
    }
  } catch (error) {
    errorHandler(new Error(toErrorMessage(error, "Could not get your location.")));
  }

  if (runtimeInfo.native) {
    const backgroundPlugin = getBackgroundPlugin();
    if (backgroundPlugin) {
      return startBackgroundPluginTracking(backgroundPlugin, {
        onLocation: locationHandler,
        onError: errorHandler,
        watchOptions,
      });
    }

    const geolocationPlugin = getGeolocationPlugin();
    if (geolocationPlugin?.watchPosition) {
      return startNativeGeolocationWatch(geolocationPlugin, {
        onLocation: locationHandler,
        onError: errorHandler,
        watchOptions,
      });
    }
  }

  return startBrowserWatch({
    onLocation: locationHandler,
    onError: errorHandler,
    options: watchOptions,
  });
};

export const addAppStateListener = async (handler) => {
  if (typeof handler !== "function") return () => {};
  const appPlugin = getAppPlugin();
  if (isNativePlatform() && appPlugin?.addListener) {
    const listener = await appPlugin.addListener("appStateChange", (state) => {
      handler({
        isActive: Boolean(state?.isActive),
        source: "native_app_state",
      });
    });
    return async () => {
      if (listener?.remove) {
        await listener.remove();
      }
    };
  }
  return () => {};
};
