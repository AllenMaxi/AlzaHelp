import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  PhoneCall,
  LocateFixed,
  Loader2,
  MapPin,
  MapPinned,
  Navigation as NavigationIcon,
  Plus,
  Route,
  Share2,
  Shield,
  ShieldAlert,
  Trash2,
  Clock,
  CheckCircle2
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { destinationsApi, navigationApi, safetyApi } from "@/services/api";
import { toast } from "sonner";

const toRad = (value) => (value * Math.PI) / 180;

const distanceKm = (a, b) => {
  const earthKm = 6371;
  const dLat = toRad(b.latitude - a.latitude);
  const dLon = toRad(b.longitude - a.longitude);
  const lat1 = toRad(a.latitude);
  const lat2 = toRad(b.latitude);

  const h =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.sin(dLon / 2) * Math.sin(dLon / 2) * Math.cos(lat1) * Math.cos(lat2);
  return 2 * earthKm * Math.asin(Math.sqrt(h));
};

export const NavigationSection = ({ destinations = [], onRefresh, loading }) => {
  const [destinationDialogOpen, setDestinationDialogOpen] = useState(false);
  const [zoneDialogOpen, setZoneDialogOpen] = useState(false);
  const [savingDestination, setSavingDestination] = useState(false);
  const [savingZone, setSavingZone] = useState(false);
  const [deletingDestination, setDeletingDestination] = useState(null);
  const [deletingZone, setDeletingZone] = useState(null);
  const [locating, setLocating] = useState(false);
  const [geoError, setGeoError] = useState("");
  const [currentLocation, setCurrentLocation] = useState(null);

  const [safeZones, setSafeZones] = useState([]);
  const [safetyAlerts, setSafetyAlerts] = useState([]);
  const [emergencyContacts, setEmergencyContacts] = useState([]);
  const [fallEvents, setFallEvents] = useState([]);
  const [pingingSafety, setPingingSafety] = useState(false);
  const [sharingLocation, setSharingLocation] = useState(false);
  const [triggeringSOS, setTriggeringSOS] = useState(false);
  const [reportingFall, setReportingFall] = useState(false);
  const [savingContact, setSavingContact] = useState(false);
  const [deletingContactId, setDeletingContactId] = useState(null);
  const [fallWatchEnabled, setFallWatchEnabled] = useState(false);
  const [escalationRules, setEscalationRules] = useState([]);
  const [escalationHistory, setEscalationHistory] = useState([]);
  const [savingEscalationRule, setSavingEscalationRule] = useState(false);
  const [runningEscalations, setRunningEscalations] = useState(false);
  const [escalationForm, setEscalationForm] = useState({
    event_type: "geofence_exit",
    min_severity: "high",
    intervals_csv: "5,15,30",
    enabled: true
  });
  const lastPingRef = useRef(0);
  const lastFallReportRef = useRef(0);

  const [guidanceDestinationId, setGuidanceDestinationId] = useState(null);
  const [guidanceLoading, setGuidanceLoading] = useState(false);
  const [guidanceResult, setGuidanceResult] = useState(null);

  const [newDestination, setNewDestination] = useState({
    name: "",
    address: "",
    visit_time: "",
    notes: "",
    latitude: "",
    longitude: ""
  });

  const [newZone, setNewZone] = useState({
    name: "",
    center_latitude: "",
    center_longitude: "",
    radius_meters: "500"
  });

  const [newEmergencyContact, setNewEmergencyContact] = useState({
    name: "",
    relationship: "caregiver",
    phone: "",
    is_primary: false,
    notes: ""
  });

  const loadSafetyData = useCallback(async () => {
    try {
      const [zones, alerts, contacts, falls, rules, history] = await Promise.all([
        safetyApi.getZones().catch(() => []),
        safetyApi.getAlerts(null, true).catch(() => []),
        safetyApi.getEmergencyContacts().catch(() => []),
        safetyApi.getFallEvents(null, 15).catch(() => []),
        safetyApi.getEscalationRules().catch(() => []),
        safetyApi.getEscalationHistory(null, 20).catch(() => [])
      ]);
      setSafeZones(zones || []);
      setSafetyAlerts(alerts || []);
      setEmergencyContacts(contacts || []);
      setFallEvents(falls || []);
      setEscalationRules(rules || []);
      setEscalationHistory(history || []);
    } catch (error) {
      console.error("Failed to load safety data", error);
    }
  }, []);

  const reportFall = useCallback(
    async (detectedBy = "manual", confidence = null, notes = "") => {
      setReportingFall(true);
      try {
        const payload = {
          detected_by: detectedBy,
          severity: "high",
          confidence,
          notes: notes || null,
          latitude: currentLocation?.latitude ?? null,
          longitude: currentLocation?.longitude ?? null
        };
        await safetyApi.reportFallEvent(payload);
        toast.error("Fall event sent to caregivers.");
        loadSafetyData();
      } catch (error) {
        if (detectedBy === "device_motion") return;
        toast.error("Could not send fall event.");
      } finally {
        setReportingFall(false);
      }
    },
    [currentLocation, loadSafetyData]
  );

  const runEscalationCycle = useCallback(
    async (silent = true) => {
      setRunningEscalations(true);
      try {
        const result = await safetyApi.runEscalations(null, 20);
        if ((result?.processed_count || 0) > 0) {
          if (!silent) {
            toast.error(`Escalated ${result.processed_count} unattended safety alert(s).`);
          }
          loadSafetyData();
        }
      } catch (error) {
        if (!silent) {
          toast.error("Could not run escalation cycle.");
        }
      } finally {
        setRunningEscalations(false);
      }
    },
    [loadSafetyData]
  );

  const refreshLocation = () => {
    if (!navigator.geolocation) {
      setGeoError("This browser does not support GPS location.");
      return;
    }

    setLocating(true);
    setGeoError("");
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setCurrentLocation({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          accuracy: position.coords.accuracy,
          updatedAt: new Date().toISOString()
        });
        setLocating(false);
      },
      (error) => {
        setLocating(false);
        setGeoError(error.message || "Could not get your location.");
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 30000 }
    );
  };

  useEffect(() => {
    refreshLocation();
    loadSafetyData();
    runEscalationCycle(true);
  }, [loadSafetyData, runEscalationCycle]);

  useEffect(() => {
    const timer = setInterval(() => {
      runEscalationCycle(true);
    }, 60000);
    return () => clearInterval(timer);
  }, [runEscalationCycle]);

  useEffect(() => {
    if (!navigator.geolocation) return undefined;

    const watcherId = navigator.geolocation.watchPosition(
      (position) => {
        const nextLocation = {
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          accuracy: position.coords.accuracy,
          updatedAt: new Date().toISOString()
        };
        setCurrentLocation(nextLocation);

        const now = Date.now();
        if (now - lastPingRef.current < 30000) return;
        lastPingRef.current = now;

        setPingingSafety(true);
        safetyApi
          .pingLocation(nextLocation.latitude, nextLocation.longitude)
          .then((result) => {
            const alerts = result?.new_alerts || [];
            if (alerts.length > 0) {
              toast.error(`Safety alert: ${alerts[0].message}`);
              loadSafetyData();
            }
          })
          .catch(() => {
            // No-op: passive monitoring should not spam errors.
          })
          .finally(() => setPingingSafety(false));
      },
      () => {
        // Passive watcher error ignored; manual refresh still available.
      },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 15000 }
    );

    return () => navigator.geolocation.clearWatch(watcherId);
  }, [loadSafetyData]);

  useEffect(() => {
    if (!fallWatchEnabled || typeof window === "undefined" || !window.addEventListener) {
      return undefined;
    }

    const onDeviceMotion = (event) => {
      const acc = event?.accelerationIncludingGravity;
      if (!acc) return;
      const magnitude = Math.sqrt(
        Math.pow(acc.x || 0, 2) + Math.pow(acc.y || 0, 2) + Math.pow(acc.z || 0, 2)
      );
      const now = Date.now();
      if (magnitude < 26 || now - lastFallReportRef.current < 120000) return;
      lastFallReportRef.current = now;
      reportFall("device_motion", 0.55, "Possible abrupt movement detected by device motion.");
    };

    window.addEventListener("devicemotion", onDeviceMotion);
    return () => window.removeEventListener("devicemotion", onDeviceMotion);
  }, [fallWatchEnabled, reportFall]);

  const addDestination = async () => {
    if (!newDestination.name.trim() || !newDestination.address.trim()) {
      toast.error("Please enter destination name and address.");
      return;
    }

    const latitude =
      newDestination.latitude === "" ? null : Number(newDestination.latitude);
    const longitude =
      newDestination.longitude === "" ? null : Number(newDestination.longitude);

    if (
      (latitude !== null && Number.isNaN(latitude)) ||
      (longitude !== null && Number.isNaN(longitude))
    ) {
      toast.error("Latitude and longitude must be valid numbers.");
      return;
    }

    setSavingDestination(true);
    try {
      await destinationsApi.create({
        name: newDestination.name.trim(),
        address: newDestination.address.trim(),
        visit_time: newDestination.visit_time.trim() || null,
        notes: newDestination.notes.trim() || null,
        latitude,
        longitude
      });
      toast.success("Destination added.");
      setNewDestination({
        name: "",
        address: "",
        visit_time: "",
        notes: "",
        latitude: "",
        longitude: ""
      });
      setDestinationDialogOpen(false);
      onRefresh();
    } catch (error) {
      toast.error("Could not save destination.");
    } finally {
      setSavingDestination(false);
    }
  };

  const addSafeZone = async () => {
    if (!newZone.name.trim() || !newZone.center_latitude || !newZone.center_longitude) {
      toast.error("Please complete zone name and coordinates.");
      return;
    }

    const radius = Number(newZone.radius_meters || 500);
    const centerLatitude = Number(newZone.center_latitude);
    const centerLongitude = Number(newZone.center_longitude);

    if (
      Number.isNaN(centerLatitude) ||
      Number.isNaN(centerLongitude) ||
      Number.isNaN(radius)
    ) {
      toast.error("Zone coordinates and radius must be valid numbers.");
      return;
    }

    setSavingZone(true);
    try {
      await safetyApi.createZone({
        name: newZone.name.trim(),
        center_latitude: centerLatitude,
        center_longitude: centerLongitude,
        radius_meters: Math.max(100, Math.round(radius))
      });
      toast.success("Safe zone created.");
      setNewZone({ name: "", center_latitude: "", center_longitude: "", radius_meters: "500" });
      setZoneDialogOpen(false);
      loadSafetyData();
    } catch (error) {
      toast.error("Could not create safe zone.");
    } finally {
      setSavingZone(false);
    }
  };

  const removeDestination = async (destinationId) => {
    setDeletingDestination(destinationId);
    try {
      await destinationsApi.delete(destinationId);
      toast.success("Destination removed.");
      if (guidanceDestinationId === destinationId) {
        setGuidanceDestinationId(null);
        setGuidanceResult(null);
      }
      onRefresh();
    } catch (error) {
      toast.error("Could not remove destination.");
    } finally {
      setDeletingDestination(null);
    }
  };

  const removeZone = async (zoneId) => {
    setDeletingZone(zoneId);
    try {
      await safetyApi.deleteZone(zoneId);
      toast.success("Safe zone removed.");
      loadSafetyData();
    } catch (error) {
      toast.error("Could not remove safe zone.");
    } finally {
      setDeletingZone(null);
    }
  };

  const addEmergencyContact = async () => {
    if (!newEmergencyContact.name.trim() || !newEmergencyContact.phone.trim()) {
      toast.error("Contact name and phone are required.");
      return;
    }
    setSavingContact(true);
    try {
      await safetyApi.createEmergencyContact({
        name: newEmergencyContact.name.trim(),
        relationship: newEmergencyContact.relationship.trim() || "caregiver",
        phone: newEmergencyContact.phone.trim(),
        is_primary: Boolean(newEmergencyContact.is_primary),
        receive_call: true,
        receive_sms: true,
        notes: newEmergencyContact.notes.trim() || null
      });
      toast.success("Emergency contact saved.");
      setNewEmergencyContact({
        name: "",
        relationship: "caregiver",
        phone: "",
        is_primary: false,
        notes: ""
      });
      loadSafetyData();
    } catch (error) {
      toast.error("Could not save emergency contact.");
    } finally {
      setSavingContact(false);
    }
  };

  const removeEmergencyContact = async (contactId) => {
    setDeletingContactId(contactId);
    try {
      await safetyApi.deleteEmergencyContact(contactId);
      toast.success("Emergency contact removed.");
      loadSafetyData();
    } catch (error) {
      toast.error("Could not remove emergency contact.");
    } finally {
      setDeletingContactId(null);
    }
  };

  const saveEscalationRule = async () => {
    const intervals = escalationForm.intervals_csv
      .split(",")
      .map((value) => Number(value.trim()))
      .filter((value) => Number.isFinite(value) && value >= 1);
    if (intervals.length === 0) {
      toast.error("Add at least one escalation interval in minutes.");
      return;
    }

    setSavingEscalationRule(true);
    try {
      await safetyApi.upsertEscalationRule({
        event_type: escalationForm.event_type,
        min_severity: escalationForm.min_severity,
        intervals_minutes: intervals,
        enabled: Boolean(escalationForm.enabled)
      });
      toast.success("Escalation rule saved.");
      loadSafetyData();
    } catch (error) {
      toast.error(error.message || "Could not save escalation rule.");
    } finally {
      setSavingEscalationRule(false);
    }
  };

  const shareCurrentLocation = async () => {
    if (!currentLocation) {
      toast.error("Location is not available yet.");
      return;
    }
    setSharingLocation(true);
    try {
      await safetyApi.shareLocation(
        currentLocation.latitude,
        currentLocation.longitude,
        "manual_share"
      );
      toast.success("Location shared with caregiver contacts.");
    } catch (error) {
      toast.error("Could not share location.");
    } finally {
      setSharingLocation(false);
    }
  };

  const triggerSOS = async () => {
    setTriggeringSOS(true);
    try {
      const result = await safetyApi.triggerSOS({
        latitude: currentLocation?.latitude ?? null,
        longitude: currentLocation?.longitude ?? null,
        message: "Patient requested emergency assistance.",
        auto_call_primary: true
      });
      toast.error("SOS sent. Caregivers were notified.");
      loadSafetyData();

      if (result?.dial_uri && window.confirm("Call primary emergency contact now?")) {
        window.location.href = result.dial_uri;
      }
    } catch (error) {
      toast.error(error?.message || "Could not trigger SOS.");
    } finally {
      setTriggeringSOS(false);
    }
  };

  const openRoute = (destination) => {
    const origin = currentLocation
      ? `${currentLocation.latitude},${currentLocation.longitude}`
      : null;

    const destinationParam =
      destination.latitude != null && destination.longitude != null
        ? `${destination.latitude},${destination.longitude}`
        : destination.address;

    const url = origin
      ? `https://www.google.com/maps/dir/?api=1&origin=${encodeURIComponent(origin)}&destination=${encodeURIComponent(destinationParam)}&travelmode=walking`
      : `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(destinationParam)}`;

    window.open(url, "_blank", "noopener,noreferrer");
  };

  const generateGuidance = async (destination) => {
    if (!currentLocation) {
      toast.error("Current location is required to generate guidance.");
      return;
    }

    setGuidanceDestinationId(destination.id);
    setGuidanceLoading(true);
    try {
      const result = await navigationApi.getGuide(
        destination.id,
        currentLocation.latitude,
        currentLocation.longitude
      );
      setGuidanceResult(result);
    } catch (error) {
      toast.error("Could not generate guidance right now.");
    } finally {
      setGuidanceLoading(false);
    }
  };

  const acknowledgeAlert = async (alertId) => {
    try {
      await safetyApi.acknowledgeAlert(alertId);
      loadSafetyData();
    } catch (error) {
      toast.error("Could not acknowledge alert.");
    }
  };

  const decoratedDestinations = useMemo(() => {
    return destinations.map((destination) => {
      let distance = null;
      if (
        currentLocation &&
        destination.latitude != null &&
        destination.longitude != null
      ) {
        distance = distanceKm(currentLocation, destination);
      }
      return { ...destination, distance };
    });
  }, [destinations, currentLocation]);

  return (
    <section className="py-8 sm:py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-4">
            <Route className="h-5 w-5 text-primary" />
            <span className="text-base font-medium text-primary">GPS Guidance</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-foreground mb-2">
            Where To Go
          </h2>
          <p className="text-accessible text-muted-foreground">
            In-app route guidance, safe zones, SOS safety actions, and caregiver alerts.
          </p>
        </div>

        <div className="grid gap-4 lg:grid-cols-4 mb-8">
          <Card className="border-2 border-border lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <LocateFixed className="h-5 w-5 text-primary" />
                Current Location
                {pingingSafety && <Loader2 className="h-4 w-4 animate-spin text-primary" />}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {currentLocation ? (
                <>
                  <p className="text-sm text-muted-foreground">
                    {currentLocation.latitude.toFixed(5)}, {currentLocation.longitude.toFixed(5)}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Accuracy: {Math.round(currentLocation.accuracy)} meters
                  </p>
                </>
              ) : (
                <p className="text-sm text-muted-foreground">Location not available yet.</p>
              )}
              {geoError && <p className="text-sm text-destructive">{geoError}</p>}
              <Button variant="outline" onClick={refreshLocation} disabled={locating} className="gap-2">
                {locating ? <Loader2 className="h-4 w-4 animate-spin" /> : <LocateFixed className="h-4 w-4" />}
                Refresh Location
              </Button>
              <Button
                variant="outline"
                onClick={shareCurrentLocation}
                disabled={!currentLocation || sharingLocation}
                className="gap-2"
              >
                {sharingLocation ? <Loader2 className="h-4 w-4 animate-spin" /> : <Share2 className="h-4 w-4" />}
                Share My Location
              </Button>
            </CardContent>
          </Card>

          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5 text-primary" />
                Safety Alerts
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {safetyAlerts.length === 0 ? (
                <p className="text-sm text-muted-foreground">No active geofence alerts.</p>
              ) : (
                safetyAlerts.slice(0, 3).map((alert) => (
                  <div key={alert.id} className="rounded-lg border border-destructive/40 p-2 bg-destructive/10">
                    <p className="text-xs font-medium text-destructive">{alert.message}</p>
                    <Button
                      variant="outline"
                      size="sm"
                      className="mt-2"
                      onClick={() => acknowledgeAlert(alert.id)}
                    >
                      <CheckCircle2 className="h-3 w-3 mr-1" /> Acknowledge
                    </Button>
                  </div>
                ))
              )}
            </CardContent>
          </Card>

          <Card className="border-2 border-destructive/40 bg-destructive/5">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-destructive">
                <PhoneCall className="h-5 w-5" />
                Emergency
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button
                variant="destructive"
                className="w-full gap-2"
                onClick={triggerSOS}
                disabled={triggeringSOS}
              >
                {triggeringSOS ? <Loader2 className="h-4 w-4 animate-spin" /> : <AlertTriangle className="h-4 w-4" />}
                SOS: Call Caregiver
              </Button>

              <Button
                variant="outline"
                className="w-full gap-2"
                onClick={() => reportFall("manual", 1.0, "Patient manually reported a fall.")}
                disabled={reportingFall}
              >
                {reportingFall ? <Loader2 className="h-4 w-4 animate-spin" /> : <Activity className="h-4 w-4" />}
                Report Fall
              </Button>

              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={fallWatchEnabled}
                  onChange={(e) => setFallWatchEnabled(e.target.checked)}
                />
                Enable motion-based fall watch
              </label>
            </CardContent>
          </Card>
        </div>

        <div className="grid gap-4 md:grid-cols-2 mb-8">
          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MapPinned className="h-5 w-5 text-primary" />
                Add Destination
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Dialog open={destinationDialogOpen} onOpenChange={setDestinationDialogOpen}>
                <DialogTrigger asChild>
                  <Button variant="accessible" className="gap-2 w-full">
                    <Plus className="h-5 w-5" />
                    New Destination
                  </Button>
                </DialogTrigger>
                <DialogContent className="max-w-lg">
                  <DialogHeader>
                    <DialogTitle className="text-2xl font-display">Add Destination</DialogTitle>
                  </DialogHeader>

                  <div className="space-y-4 mt-3">
                    <div>
                      <Label htmlFor="dest-name">Name</Label>
                      <Input
                        id="dest-name"
                        value={newDestination.name}
                        onChange={(e) =>
                          setNewDestination((prev) => ({ ...prev, name: e.target.value }))
                        }
                        placeholder="e.g., Therapy Clinic"
                        className="mt-2"
                      />
                    </div>

                    <div>
                      <Label htmlFor="dest-address">Address</Label>
                      <Input
                        id="dest-address"
                        value={newDestination.address}
                        onChange={(e) =>
                          setNewDestination((prev) => ({ ...prev, address: e.target.value }))
                        }
                        placeholder="Street, city"
                        className="mt-2"
                      />
                    </div>

                    <div>
                      <Label htmlFor="dest-time">Visit Time (optional)</Label>
                      <Input
                        id="dest-time"
                        value={newDestination.visit_time}
                        onChange={(e) =>
                          setNewDestination((prev) => ({ ...prev, visit_time: e.target.value }))
                        }
                        placeholder="e.g., Today 3:00 PM"
                        className="mt-2"
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <Label htmlFor="dest-lat">Latitude (optional)</Label>
                        <Input
                          id="dest-lat"
                          value={newDestination.latitude}
                          onChange={(e) =>
                            setNewDestination((prev) => ({ ...prev, latitude: e.target.value }))
                          }
                          placeholder="40.7128"
                          className="mt-2"
                        />
                      </div>
                      <div>
                        <Label htmlFor="dest-lng">Longitude (optional)</Label>
                        <Input
                          id="dest-lng"
                          value={newDestination.longitude}
                          onChange={(e) =>
                            setNewDestination((prev) => ({ ...prev, longitude: e.target.value }))
                          }
                          placeholder="-74.0060"
                          className="mt-2"
                        />
                      </div>
                    </div>

                    <div>
                      <Label htmlFor="dest-notes">Notes (optional)</Label>
                      <Textarea
                        id="dest-notes"
                        value={newDestination.notes}
                        onChange={(e) =>
                          setNewDestination((prev) => ({ ...prev, notes: e.target.value }))
                        }
                        placeholder="Add any guidance for this trip."
                        className="mt-2"
                      />
                    </div>

                    <Button className="w-full gap-2" variant="accessible" onClick={addDestination} disabled={savingDestination}>
                      {savingDestination ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
                      Save Destination
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>
            </CardContent>
          </Card>

          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5 text-primary" />
                Safe Zones
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Dialog open={zoneDialogOpen} onOpenChange={setZoneDialogOpen}>
                <DialogTrigger asChild>
                  <Button variant="outline" className="w-full gap-2">
                    <Plus className="h-4 w-4" /> New Safe Zone
                  </Button>
                </DialogTrigger>
                <DialogContent className="max-w-lg">
                  <DialogHeader>
                    <DialogTitle className="text-2xl font-display">Create Safe Zone</DialogTitle>
                  </DialogHeader>
                  <div className="space-y-4 mt-3">
                    <div>
                      <Label htmlFor="zone-name">Zone Name</Label>
                      <Input
                        id="zone-name"
                        value={newZone.name}
                        onChange={(e) => setNewZone((prev) => ({ ...prev, name: e.target.value }))}
                        placeholder="e.g., Home Area"
                        className="mt-2"
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <Label htmlFor="zone-lat">Center Latitude</Label>
                        <Input
                          id="zone-lat"
                          value={newZone.center_latitude}
                          onChange={(e) => setNewZone((prev) => ({ ...prev, center_latitude: e.target.value }))}
                          placeholder={currentLocation ? currentLocation.latitude.toFixed(5) : "40.7128"}
                          className="mt-2"
                        />
                      </div>
                      <div>
                        <Label htmlFor="zone-lng">Center Longitude</Label>
                        <Input
                          id="zone-lng"
                          value={newZone.center_longitude}
                          onChange={(e) => setNewZone((prev) => ({ ...prev, center_longitude: e.target.value }))}
                          placeholder={currentLocation ? currentLocation.longitude.toFixed(5) : "-74.0060"}
                          className="mt-2"
                        />
                      </div>
                    </div>
                    <div>
                      <Label htmlFor="zone-radius">Radius (meters)</Label>
                      <Input
                        id="zone-radius"
                        value={newZone.radius_meters}
                        onChange={(e) => setNewZone((prev) => ({ ...prev, radius_meters: e.target.value }))}
                        placeholder="500"
                        className="mt-2"
                      />
                    </div>
                    <Button className="w-full" variant="accessible" onClick={addSafeZone} disabled={savingZone}>
                      {savingZone ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                      Save Safe Zone
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>

              {safeZones.length === 0 ? (
                <p className="text-sm text-muted-foreground">No zones yet.</p>
              ) : (
                <div className="space-y-2">
                  {safeZones.map((zone) => (
                    <div key={zone.id} className="rounded-lg border p-2 flex items-center justify-between gap-2">
                      <div>
                        <p className="text-sm font-medium">{zone.name}</p>
                        <p className="text-xs text-muted-foreground">Radius: {zone.radius_meters}m</p>
                      </div>
                      <Button
                        variant="destructive"
                        size="icon"
                        onClick={() => removeZone(zone.id)}
                        disabled={deletingZone === zone.id}
                      >
                        {deletingZone === zone.id ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Trash2 className="h-4 w-4" />
                        )}
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="grid gap-4 lg:grid-cols-2 mb-8">
          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <PhoneCall className="h-5 w-5 text-primary" />
                Emergency Contacts
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <Input
                  placeholder="Contact name"
                  value={newEmergencyContact.name}
                  onChange={(e) => setNewEmergencyContact((prev) => ({ ...prev, name: e.target.value }))}
                />
                <Input
                  placeholder="Relationship"
                  value={newEmergencyContact.relationship}
                  onChange={(e) => setNewEmergencyContact((prev) => ({ ...prev, relationship: e.target.value }))}
                />
              </div>
              <Input
                placeholder="Phone"
                value={newEmergencyContact.phone}
                onChange={(e) => setNewEmergencyContact((prev) => ({ ...prev, phone: e.target.value }))}
              />
              <Textarea
                placeholder="Notes (optional)"
                value={newEmergencyContact.notes}
                onChange={(e) => setNewEmergencyContact((prev) => ({ ...prev, notes: e.target.value }))}
              />
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={newEmergencyContact.is_primary}
                  onChange={(e) => setNewEmergencyContact((prev) => ({ ...prev, is_primary: e.target.checked }))}
                />
                Primary contact
              </label>
              <Button onClick={addEmergencyContact} disabled={savingContact} className="gap-2">
                {savingContact ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
                Save Emergency Contact
              </Button>

              {emergencyContacts.length === 0 ? (
                <p className="text-sm text-muted-foreground">No emergency contacts yet.</p>
              ) : (
                <div className="space-y-2">
                  {emergencyContacts.map((contact) => (
                    <div key={contact.id} className="rounded-lg border p-3 flex items-center justify-between gap-3">
                      <div>
                        <p className="font-medium text-sm">
                          {contact.name} {contact.is_primary ? "(Primary)" : ""}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {contact.relationship} - {contact.phone}
                        </p>
                      </div>
                      <Button
                        variant="destructive"
                        size="icon"
                        onClick={() => removeEmergencyContact(contact.id)}
                        disabled={deletingContactId === contact.id}
                      >
                        {deletingContactId === contact.id ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Trash2 className="h-4 w-4" />
                        )}
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-primary" />
                Recent Fall Events
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {fallEvents.length === 0 ? (
                <p className="text-sm text-muted-foreground">No fall events reported.</p>
              ) : (
                fallEvents.slice(0, 6).map((event) => (
                  <div key={event.id} className="rounded-lg border p-3">
                    <p className="text-sm font-medium capitalize">
                      {event.detected_by?.replace("_", " ") || "manual"} - {event.severity || "high"}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {new Date(event.detected_at).toLocaleString()}
                    </p>
                    {event.notes ? <p className="text-xs mt-1">{event.notes}</p> : null}
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        </div>

        <div className="grid gap-4 lg:grid-cols-2 mb-8">
          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <ShieldAlert className="h-5 w-5 text-primary" />
                Escalation Rules
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <Label>Event</Label>
                  <select
                    className="w-full mt-2 rounded-md border border-input bg-background px-3 py-2"
                    value={escalationForm.event_type}
                    onChange={(e) => setEscalationForm((prev) => ({ ...prev, event_type: e.target.value }))}
                  >
                    <option value="geofence_exit">Geofence Exit</option>
                    <option value="sos_trigger">SOS Trigger</option>
                    <option value="fall_detected">Fall Detected</option>
                    <option value="missed_medication_dose">Missed Dose</option>
                    <option value="all">All Events</option>
                  </select>
                </div>
                <div>
                  <Label>Min severity</Label>
                  <select
                    className="w-full mt-2 rounded-md border border-input bg-background px-3 py-2"
                    value={escalationForm.min_severity}
                    onChange={(e) => setEscalationForm((prev) => ({ ...prev, min_severity: e.target.value }))}
                  >
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="critical">Critical</option>
                  </select>
                </div>
              </div>
              <div>
                <Label>Intervals in minutes (comma-separated)</Label>
                <Input
                  value={escalationForm.intervals_csv}
                  onChange={(e) => setEscalationForm((prev) => ({ ...prev, intervals_csv: e.target.value }))}
                  placeholder="5,15,30"
                />
              </div>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={escalationForm.enabled}
                  onChange={(e) => setEscalationForm((prev) => ({ ...prev, enabled: e.target.checked }))}
                />
                Rule enabled
              </label>
              <div className="flex gap-2">
                <Button onClick={saveEscalationRule} disabled={savingEscalationRule} className="gap-2">
                  {savingEscalationRule ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}
                  Save Rule
                </Button>
                <Button variant="outline" onClick={() => runEscalationCycle(false)} disabled={runningEscalations} className="gap-2">
                  {runningEscalations ? <Loader2 className="h-4 w-4 animate-spin" /> : <Activity className="h-4 w-4" />}
                  Run Escalation Cycle
                </Button>
              </div>

              {escalationRules.length > 0 && (
                <div className="space-y-2">
                  {escalationRules.map((rule) => (
                    <div key={rule.id} className="rounded-lg border p-2">
                      <p className="text-sm font-medium capitalize">{(rule.event_type || "all").replace(/_/g, " ")}</p>
                      <p className="text-xs text-muted-foreground">
                        Min severity: {rule.min_severity} · Intervals: {(rule.intervals_minutes || []).join(", ")} min · {rule.enabled ? "Enabled" : "Disabled"}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle>Recent Escalation History</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {escalationHistory.length === 0 ? (
                <p className="text-sm text-muted-foreground">No escalation events yet.</p>
              ) : (
                escalationHistory.slice(0, 8).map((event) => (
                  <div key={event.id} className="rounded-lg border p-2">
                    <p className="text-sm font-medium capitalize">
                      {(event.event_type || "alert").replace(/_/g, " ")} · {event.action}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Stage: {event.stage ?? 0} · {new Date(event.created_at).toLocaleString()}
                    </p>
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        </div>

        {loading && (
          <div className="flex justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        )}

        {!loading && decoratedDestinations.length === 0 && (
          <Card className="border-2 border-border">
            <CardContent className="p-8 text-center">
              <MapPinned className="h-10 w-10 text-muted-foreground mx-auto mb-3" />
              <h3 className="text-xl font-semibold mb-2">No destinations yet</h3>
              <p className="text-muted-foreground">
                Add places like clinic, pharmacy, or family home to get directions quickly.
              </p>
            </CardContent>
          </Card>
        )}

        {!loading && decoratedDestinations.length > 0 && (
          <div className="grid gap-4">
            {decoratedDestinations.map((destination) => (
              <Card key={destination.id} className="border-2 border-border shadow-soft">
                <CardContent className="p-5">
                  <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
                    <div className="space-y-2">
                      <h3 className="text-xl font-semibold">{destination.name}</h3>
                      <p className="text-muted-foreground flex items-center gap-2">
                        <MapPin className="h-4 w-4" />
                        {destination.address}
                      </p>
                      {destination.visit_time && (
                        <p className="text-muted-foreground flex items-center gap-2">
                          <Clock className="h-4 w-4" />
                          {destination.visit_time}
                        </p>
                      )}
                      {destination.distance != null && (
                        <Badge variant="secondary">{destination.distance.toFixed(1)} km away</Badge>
                      )}
                      {destination.notes && (
                        <p className="text-sm text-muted-foreground">{destination.notes}</p>
                      )}
                    </div>

                    <div className="flex flex-wrap gap-2">
                      <Button
                        variant="accessible"
                        className="gap-2"
                        onClick={() => openRoute(destination)}
                      >
                        <NavigationIcon className="h-4 w-4" />
                        Open Map
                      </Button>
                      <Button
                        variant="outline"
                        className="gap-2"
                        onClick={() => generateGuidance(destination)}
                        disabled={guidanceLoading && guidanceDestinationId === destination.id}
                      >
                        {guidanceLoading && guidanceDestinationId === destination.id ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Route className="h-4 w-4" />
                        )}
                        In-App Guide
                      </Button>
                      <Button
                        variant="destructive"
                        size="icon"
                        onClick={() => removeDestination(destination.id)}
                        disabled={deletingDestination === destination.id}
                      >
                        {deletingDestination === destination.id ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Trash2 className="h-4 w-4" />
                        )}
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {guidanceResult && (
          <Card className="border-2 border-primary mt-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Route className="h-5 w-5 text-primary" />
                Turn-by-Turn Guidance: {guidanceResult.destination}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2 mb-4">
                {guidanceResult.distance_meters != null && (
                  <Badge variant="secondary">Distance: {(guidanceResult.distance_meters / 1000).toFixed(2)} km</Badge>
                )}
                {guidanceResult.eta_minutes != null && (
                  <Badge variant="secondary">ETA: {guidanceResult.eta_minutes} min walk</Badge>
                )}
                {guidanceResult.direction && (
                  <Badge variant="secondary">Direction: {guidanceResult.direction}</Badge>
                )}
              </div>
              <ol className="space-y-2">
                {(guidanceResult.steps || []).map((step, index) => (
                  <li key={`${step}-${index}`} className="text-sm flex gap-2">
                    <span className="font-semibold text-primary">{index + 1}.</span>
                    <span>{step}</span>
                  </li>
                ))}
              </ol>
            </CardContent>
          </Card>
        )}

        {safetyAlerts.length > 0 && (
          <Card className="border-2 border-destructive/40 mt-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-destructive">
                <AlertTriangle className="h-5 w-5" />
                Active Safety Alerts
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {safetyAlerts.map((alert) => (
                <div key={alert.id} className="rounded-lg border border-destructive/40 p-3 bg-destructive/10 flex justify-between gap-3">
                  <div>
                    <p className="font-medium text-sm">{alert.message}</p>
                    <p className="text-xs text-muted-foreground">Triggered: {new Date(alert.triggered_at).toLocaleString()}</p>
                  </div>
                  <Button variant="outline" size="sm" onClick={() => acknowledgeAlert(alert.id)}>
                    Acknowledge
                  </Button>
                </div>
              ))}
            </CardContent>
          </Card>
        )}
      </div>
    </section>
  );
};

export default NavigationSection;
