package com.alzahelp.mobile.location;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Looper;
import android.provider.Settings;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;

import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import com.getcapacitor.annotation.Permission;

import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationCallback;
import com.google.android.gms.location.LocationRequest;
import com.google.android.gms.location.LocationResult;
import com.google.android.gms.location.LocationServices;
import com.google.android.gms.location.Priority;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Map;
import java.util.TimeZone;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

@CapacitorPlugin(
    name = "AlzaBackgroundLocation",
    permissions = {
        @Permission(
            strings = { Manifest.permission.ACCESS_FINE_LOCATION },
            alias = "location"
        ),
        @Permission(
            strings = { Manifest.permission.ACCESS_COARSE_LOCATION },
            alias = "coarseLocation"
        )
    }
)
public class AlzaBackgroundLocationPlugin extends Plugin {

    private static final String TAG = "AlzaBackgroundLocation";

    private static final int DEFAULT_INTERVAL_MS = 30000;
    private static final int DEFAULT_DISTANCE_FILTER_METERS = 25;

    private FusedLocationProviderClient fusedClient;
    private LocationCallback sharedLocationCallback;

    private final ConcurrentHashMap<String, WatcherEntry> watchers = new ConcurrentHashMap<>();

    private int activeIntervalMs = DEFAULT_INTERVAL_MS;
    private int activeDistanceFilter = DEFAULT_DISTANCE_FILTER_METERS;
    private int activePriority = Priority.PRIORITY_HIGH_ACCURACY;
    private boolean flpActive = false;

    // Pending call for background permission flow
    private PluginCall pendingPermissionCall;

    private static class WatcherEntry {
        final PluginCall call;
        final int intervalMs;
        final int distanceFilter;
        final int priority;

        WatcherEntry(PluginCall call, int intervalMs, int distanceFilter, int priority) {
            this.call = call;
            this.intervalMs = intervalMs;
            this.distanceFilter = distanceFilter;
            this.priority = priority;
        }
    }

    @Override
    public void load() {
        fusedClient = LocationServices.getFusedLocationProviderClient(getActivity());
        sharedLocationCallback = new LocationCallback() {
            @Override
            public void onLocationResult(@NonNull LocationResult locationResult) {
                if (locationResult.getLastLocation() == null) return;
                for (android.location.Location location : locationResult.getLocations()) {
                    JSObject data = locationToJS(location);
                    for (WatcherEntry entry : watchers.values()) {
                        entry.call.resolve(data);
                    }
                }
            }
        };
    }

    // ─── requestPermissions ───────────────────────────────────────

    @PluginMethod()
    public void requestPermissions(PluginCall call) {
        // Step 1: Check fine location
        if (!hasFineLocationPermission()) {
            pendingPermissionCall = call;
            requestPermissionForAlias("location", call, "handleFineLocationResult");
            return;
        }

        // Fine already granted — check background
        resolveOrUpgradeBackground(call);
    }

    @PluginMethod()
    public void handleFineLocationResult(PluginCall call) {
        PluginCall original = pendingPermissionCall;
        pendingPermissionCall = null;
        if (original == null) {
            call.resolve(buildPermissionState());
            return;
        }

        if (!hasFineLocationPermission()) {
            original.resolve(buildPermissionState());
            return;
        }

        // Fine granted — check background
        resolveOrUpgradeBackground(original);
    }

    private void resolveOrUpgradeBackground(PluginCall call) {
        if (hasBackgroundLocationPermission()) {
            call.resolve(buildPermissionState());
            return;
        }

        // On Android 10, we can request at runtime
        if (Build.VERSION.SDK_INT == Build.VERSION_CODES.Q) {
            ActivityCompat.requestPermissions(
                getActivity(),
                new String[]{ Manifest.permission.ACCESS_BACKGROUND_LOCATION },
                9999
            );
            call.resolve(buildPermissionState());
            return;
        }

        // Android 11+: must deep-link to settings
        try {
            Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
            intent.setData(Uri.fromParts("package", getContext().getPackageName(), null));
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            getContext().startActivity(intent);
        } catch (Exception e) {
            Log.w(TAG, "Could not open app settings", e);
        }

        // Resolve with current state — JS can show rationale
        call.resolve(buildPermissionState());
    }

    private JSObject buildPermissionState() {
        JSObject state = new JSObject();
        state.put("location", hasFineLocationPermission() ? "granted" : "denied");
        state.put("background", hasBackgroundLocationPermission() ? "granted" : "denied");
        return state;
    }

    // ─── getCurrentPosition ──────────────────────────────────────

    @PluginMethod()
    public void getCurrentPosition(PluginCall call) {
        if (!hasFineLocationPermission()) {
            call.reject("Location permission not granted");
            return;
        }

        try {
            fusedClient.getLastLocation().addOnSuccessListener(location -> {
                if (location == null) {
                    call.reject("Could not determine current location");
                    return;
                }
                call.resolve(locationToJS(location));
            }).addOnFailureListener(e -> {
                call.reject("Location request failed: " + e.getMessage());
            });
        } catch (SecurityException e) {
            call.reject("Location permission revoked");
        }
    }

    // ─── addWatcher ──────────────────────────────────────────────

    @PluginMethod(returnType = PluginMethod.RETURN_CALLBACK)
    public void addWatcher(PluginCall call) {
        call.setKeepAlive(true);

        if (!hasFineLocationPermission()) {
            JSObject err = new JSObject();
            err.put("error", "Location permission not granted");
            call.resolve(err);
            return;
        }

        String watcherId = "watcher_" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);

        int intervalMs = call.getInt("intervalMs", DEFAULT_INTERVAL_MS);
        int distanceFilter = call.getInt("minDistanceMeters", DEFAULT_DISTANCE_FILTER_METERS);
        // Map string priority from JS or use int directly
        int priority = resolvePriority(call);

        WatcherEntry entry = new WatcherEntry(call, intervalMs, distanceFilter, priority);
        watchers.put(watcherId, entry);

        // Log effective config for debugging
        Log.i(TAG, "addWatcher: id=" + watcherId
            + " interval=" + intervalMs + "ms"
            + " distance=" + distanceFilter + "m"
            + " priority=" + priority
            + " totalWatchers=" + watchers.size());

        // Start or reconfigure FLP
        reconfigureFLP();

        // Start foreground service
        String title = call.getString("backgroundTitle", "AlzaHelp location monitoring");
        String message = call.getString("backgroundMessage", "Tracking location to keep safety alerts up to date.");
        BackgroundLocationServiceController.start(getContext(), title, message);

        // Return watcher ID via initial resolve
        JSObject ret = new JSObject();
        ret.put("id", watcherId);
        call.resolve(ret);
    }

    // ─── removeWatcher ───────────────────────────────────────────

    @PluginMethod()
    public void removeWatcher(PluginCall call) {
        String watcherId = call.getString("id");
        if (watcherId == null) {
            call.reject("Missing watcher id");
            return;
        }

        WatcherEntry removed = watchers.remove(watcherId);
        if (removed != null) {
            removed.call.setKeepAlive(false);
            try { removed.call.resolve(new JSObject()); } catch (Exception ignored) {}
        }

        Log.i(TAG, "removeWatcher: id=" + watcherId + " remainingWatchers=" + watchers.size());

        if (watchers.isEmpty()) {
            stopFLP();
            BackgroundLocationServiceController.stop(getContext());
        } else {
            reconfigureFLP();
        }

        call.resolve();
    }

    // ─── FLP management ──────────────────────────────────────────

    private void reconfigureFLP() {
        // Calculate most aggressive settings from all watchers
        int bestInterval = Integer.MAX_VALUE;
        int bestDistance = Integer.MAX_VALUE;
        int bestPriority = Priority.PRIORITY_LOW_POWER;

        for (WatcherEntry w : watchers.values()) {
            bestInterval = Math.min(bestInterval, w.intervalMs);
            bestDistance = Math.min(bestDistance, w.distanceFilter);
            bestPriority = Math.min(bestPriority, w.priority); // lower = higher accuracy
        }

        if (bestInterval == activeIntervalMs
            && bestDistance == activeDistanceFilter
            && bestPriority == activePriority
            && flpActive) {
            return; // no change needed
        }

        activeIntervalMs = bestInterval;
        activeDistanceFilter = bestDistance;
        activePriority = bestPriority;

        // Stop current if active
        if (flpActive) {
            try { fusedClient.removeLocationUpdates(sharedLocationCallback); } catch (Exception ignored) {}
        }

        LocationRequest request = new LocationRequest.Builder(activePriority, activeIntervalMs)
            .setMinUpdateIntervalMillis(activeIntervalMs / 2)
            .setMinUpdateDistanceMeters(activeDistanceFilter)
            .build();

        try {
            fusedClient.requestLocationUpdates(request, sharedLocationCallback, Looper.getMainLooper());
            flpActive = true;
            Log.i(TAG, "FLP configured: interval=" + activeIntervalMs
                + "ms distance=" + activeDistanceFilter
                + "m priority=" + activePriority);
        } catch (SecurityException e) {
            Log.e(TAG, "FLP permission denied", e);
            flpActive = false;
        }
    }

    private void stopFLP() {
        if (flpActive) {
            try { fusedClient.removeLocationUpdates(sharedLocationCallback); } catch (Exception ignored) {}
            flpActive = false;
            Log.i(TAG, "FLP stopped");
        }
    }

    // ─── Resume: re-check background permission ──────────────────

    @Override
    protected void handleOnResume() {
        super.handleOnResume();
        // Re-check background permission (user may have changed it in settings)
        if (!watchers.isEmpty() && hasBackgroundLocationPermission()) {
            Log.i(TAG, "Background location permission confirmed on resume");
        }
    }

    @Override
    protected void handleOnDestroy() {
        super.handleOnDestroy();
        stopFLP();
        watchers.clear();
    }

    // ─── Helpers ─────────────────────────────────────────────────

    private boolean hasFineLocationPermission() {
        return ActivityCompat.checkSelfPermission(getContext(),
            Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED;
    }

    private boolean hasBackgroundLocationPermission() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            return hasFineLocationPermission(); // pre-Android 10: fine = background
        }
        return ActivityCompat.checkSelfPermission(getContext(),
            Manifest.permission.ACCESS_BACKGROUND_LOCATION) == PackageManager.PERMISSION_GRANTED;
    }

    private int resolvePriority(PluginCall call) {
        String priorityStr = call.getString("priority", "");
        if ("balanced".equalsIgnoreCase(priorityStr)) return Priority.PRIORITY_BALANCED_POWER_ACCURACY;
        if ("low".equalsIgnoreCase(priorityStr)) return Priority.PRIORITY_LOW_POWER;
        if ("passive".equalsIgnoreCase(priorityStr)) return Priority.PRIORITY_PASSIVE;
        // Default to high accuracy for safety tracking
        return Priority.PRIORITY_HIGH_ACCURACY;
    }

    private JSObject locationToJS(android.location.Location location) {
        JSObject obj = new JSObject();
        obj.put("latitude", location.getLatitude());
        obj.put("longitude", location.getLongitude());
        obj.put("accuracy", location.hasAccuracy() ? location.getAccuracy() : null);
        obj.put("altitude", location.hasAltitude() ? location.getAltitude() : null);
        obj.put("speed", location.hasSpeed() ? location.getSpeed() : null);
        obj.put("bearing", location.hasBearing() ? location.getBearing() : null);

        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US);
        sdf.setTimeZone(TimeZone.getTimeZone("UTC"));
        obj.put("timestamp", sdf.format(new Date(location.getTime())));

        return obj;
    }
}
