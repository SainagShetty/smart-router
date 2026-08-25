// PM2 config for the shared smartrouter server on the Mac Mini.
//
//   pm2 start examples/ecosystem.config.js
//   pm2 logs smart-router
//   pm2 restart smart-router
//
// Provider keys and the optional bearer token are read from the environment, so
// they never sit in this file. Export them in the shell PM2 is started from, or
// use a pm2 env file. Point your services at http://127.0.0.1:4000/v1.
module.exports = {
  apps: [
    {
      name: "smart-router",
      // absolute path to the venv console script so PM2 doesn't need an
      // activated shell; adjust if your venv lives elsewhere.
      script: "/Users/sainagshetty/Development/smart-router/.venv/bin/smartrouter",
      args: "serve --config ./examples/router.yaml --host 127.0.0.1 --port 4000",
      interpreter: "none", // smartrouter is a console script, not a .js file
      cwd: "/Users/sainagshetty/Development/smart-router",
      env: {
        // Set these in PM2's environment (do not hardcode secrets here):
        // OPENROUTER_API_KEY: "...",
        // SMARTROUTER_API_KEY: "...",   // bearer token required of off-box callers
        //   (on-box loopback callers stay exempt unless you also set
        //   SMARTROUTER_TRUST_LOOPBACK=0 -- see docs/CONNECTING.md)
        //
        // NB: these live only in ~/.pm2/dump.pm2 once you `pm2 save`. A bare
        // `pm2 restart smart-router --update-env` from a shell that lacks them
        // will wipe them and break the router -- export them first.
        // The RENDERED config, not this repo's example. The active revision
        // lives in smartrouter.db and renders to this path; the router loads
        // the file, which is what keeps the database off the boot path -- a
        // lock there at 03:30 must never stop the gateway starting.
        SMARTROUTER_CONFIG: "/Users/sainagshetty/.config/smartrouter/router.yaml",
        // SMARTROUTER_ADMIN_TOKEN: "..."   // required by /admin/*; Caddy
        //   injects it for browsers so the token never reaches one.
      },
      autorestart: true,
      max_restarts: 10,
    },
  ],
};
