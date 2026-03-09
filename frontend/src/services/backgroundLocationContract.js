export const BACKGROUND_LOCATION_PLUGIN_NAME = "AlzaBackgroundLocation";
export const BACKGROUND_LOCATION_PLUGIN_ALIASES = [
  BACKGROUND_LOCATION_PLUGIN_NAME,
  "BackgroundGeolocation",
  "BackgroundLocation",
];

export const BACKGROUND_LOCATION_METHODS = {
  addWatcher: "addWatcher",
  removeWatcher: "removeWatcher",
  getCurrentPosition: "getCurrentPosition",
  getCurrentLocation: "getCurrentLocation",
  requestPermissions: "requestPermissions",
};

export const BACKGROUND_LOCATION_EVENTS = {
  location: "location",
  error: "error",
};

export const BACKGROUND_LOCATION_SERVICE = {
  androidChannelId: "alzahelp_location_tracking",
  androidNotificationId: 4107,
  androidActionStart: "com.alzahelp.mobile.action.START_BACKGROUND_LOCATION",
  androidActionStop: "com.alzahelp.mobile.action.STOP_BACKGROUND_LOCATION",
  androidExtraNotificationTitle: "notificationTitle",
  androidExtraNotificationText: "notificationText",
};

export const DEFAULT_BACKGROUND_LOCATION_OPTIONS = {
  requestPermissions: true,
  stale: false,
  distanceFilter: 25,
  backgroundTitle: "AlzaHelp location monitoring",
  backgroundMessage: "Tracking location to keep safety alerts up to date.",
};

/**
 * Tracking profiles — passed to addWatcher to control GPS behavior.
 * The Android plugin reads intervalMs, minDistanceMeters, and priority.
 */
export const TRACKING_PROFILES = {
  safety: {
    intervalMs: 30000,
    minDistanceMeters: 25,
    priority: 'high',
  },
  navigation: {
    intervalMs: 5000,
    minDistanceMeters: 5,
    priority: 'high',
  },
  power_save: {
    intervalMs: 120000,
    minDistanceMeters: 100,
    priority: 'balanced',
  },
};
