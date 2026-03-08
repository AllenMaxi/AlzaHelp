# Native Mobile Layer

This project now includes a native-aware location runtime in the web app:

- `/Users/maximilianoallende/Documents/memory-keeper/AlzaHelp/frontend/src/services/locationRuntime.js`
- `/Users/maximilianoallende/Documents/memory-keeper/AlzaHelp/frontend/src/context/LocationTrackingContext.jsx`
- `/Users/maximilianoallende/Documents/memory-keeper/AlzaHelp/frontend/capacitor.config.json`

The frontend will prefer the following runtime order:

1. A native background location plugin exposed through Capacitor
2. Capacitor Geolocation foreground tracking
3. Browser geolocation fallback

## Expected Native Plugin Contract

For dependable always-on geofencing, the iOS/Android layer should expose one of these patterns:

### Preferred

- `Capacitor.Plugins.AlzaBackgroundLocation.addWatcher(options, callback)`
- `Capacitor.Plugins.AlzaBackgroundLocation.removeWatcher({ id })`
- `Capacitor.Plugins.AlzaBackgroundLocation.getCurrentPosition(options)`
- `Capacitor.Plugins.AlzaBackgroundLocation.requestPermissions()`

Callback payload should look like:

```json
{
  "latitude": 40.4168,
  "longitude": -3.7038,
  "accuracy": 12,
  "timestamp": "2026-03-08T12:00:00.000Z"
}
```

### Also Supported

- `Capacitor.Plugins.BackgroundGeolocation.addWatcher(...)`
- `Capacitor.Plugins.BackgroundGeolocation.removeWatcher(...)`
- `Capacitor.Plugins.Geolocation.watchPosition(...)`
- `Capacitor.Plugins.Geolocation.getCurrentPosition(...)`

## Native Build Steps

When network/tooling access is available, generate the real shell from `/Users/maximilianoallende/Documents/memory-keeper/AlzaHelp/frontend`:

```bash
yarn add @capacitor/core @capacitor/cli @capacitor/app @capacitor/geolocation
npx cap add ios
npx cap add android
npx cap sync
```

For true background tracking, add a native background location plugin or a small custom Capacitor plugin that matches the contract above. Standard foreground geolocation alone is not enough for dependable unattended geofencing.

## Platform Notes

- iOS needs `NSLocationWhenInUseUsageDescription` and `NSLocationAlwaysAndWhenInUseUsageDescription`.
- Android needs foreground and background location permissions plus a foreground service notification.
- The backend already accepts `captured_at`, `source`, and `app_state`, so native samples can be uploaded without API changes.
