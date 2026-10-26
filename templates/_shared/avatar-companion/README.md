## Avatar Companion

A PlayCanvas avatar demo with pointer reactions, a converted standing animation, optional webcam
face-pose tracking, browser speech recognition and speech synthesis, a configurable
OpenAI-compatible chat endpoint, material tints, model replacement, a local activity feed,
web/video embeds and install/notification affordances.

### Run

```bash
npm install
npm run dev
```

The supplied avatar is loaded as a glTF container. Its file is a legacy VRM 0.x asset (the file
does not contain the VRM 1.0 `VRMC_vrm` extension), so PlayCanvas displays its standard glTF
geometry and skin but does not interpret VRM-only expressions or humanoid metadata. The model
picker accepts `.vrm` and `.glb` files that use the GLB container format. The bundled `stand3.vmd`
motion is a modified, selected-track, axis-adjusted conversion to a small JSON animation, mapped to
the avatar's humanoid bone names; it is not a general-purpose VMD importer.

The loop uses the quieter `stand2.vmd` clip. The VMD motions were created by 鈴桜 (遠州弁トークP). Their accompanying readme permits modifications
and modified redistribution, but requests that the original files not be redistributed unchanged.
Only selected, converted keyframes are included here.

### Optional browser capabilities

- **Face tracking:** starts only after permission is granted. MediaPipe Face Landmarker runs in the
  browser; its WASM and model are downloaded from jsDelivr and Google on first use. Camera frames
  are not uploaded by this app. Tracking estimates head position and orientation for a six-axis
  camera response. Use a secure context (HTTPS or localhost).
- **Speech recognition:** uses the browser's Web Speech API where available; browser support varies.
- **Voice:** uses voices installed by the browser/operating system. This is browser speech synthesis,
  not a hosted AI voice service.
- **Chat:** defaults to a local Ollama-compatible endpoint and the small
  `qwen2.5:0.5b-instruct` model. Install and run Ollama separately, enable browser CORS for the
  app's origin, then adjust the endpoint/model if needed. A remote OpenAI-compatible endpoint can
  also be configured. API keys are kept in memory only, but a browser app cannot protect a key from
  the device or network; prefer a trusted server-side proxy for production.
- **Web/video:** external sites may refuse embedding. URLs are opened in a sandboxed frame; the app
  does not proxy or download their content.
- **Install and notifications:** installation depends on browser/PWA support. Notifications are
  foreground browser notifications for local feed events; remote push requires a service worker and
  a push backend, neither of which is included.
