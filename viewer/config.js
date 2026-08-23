// Backend origin for a statically-hosted viewer.
//
// Only needed when this page is served from somewhere other than the Maestro API
// (e.g. Cloudflare Pages / Netlify while the service runs on Render). Set it to the
// API's origin, with no trailing slash:
//
//   window.MAESTRO_API = "https://maestro-xxxx.onrender.com";
//
// Left empty, the viewer calls its own origin — which is what you want when FastAPI
// is serving it at "/". FastAPI serves its own empty stub at /config.js, so this
// file only ever takes effect on a static host.
//
// A ?api=<origin> query parameter overrides this, for pointing a local copy of the
// page at a deployed backend without editing anything.
window.MAESTRO_API = "https://maestro-bb8l.onrender.com";
