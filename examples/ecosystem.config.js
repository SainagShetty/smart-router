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
        // SMARTROUTER_API_KEY: "...",   // optional bearer token for clients
        SMARTROUTER_CONFIG: "./examples/router.yaml",
      },
      autorestart: true,
      max_restarts: 10,
    },
  ],
};
