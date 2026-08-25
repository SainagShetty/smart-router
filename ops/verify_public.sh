#!/bin/bash
#
# Verify router.sainag.com from outside, as an anonymous visitor sees it.
#
# Two failures to catch, and the SECOND is the dangerous one:
#
#   1. Cloudflare Access not protecting /admin -> anyone can rewrite the routing
#      for ten services.
#   2. /v1/* reachable on this hostname -> the LLM gateway holding this
#      machine's provider keys is on the public internet. The Caddy block
#      allowlists /admin* and 404s everything else; this proves the allowlist is
#      actually in force, not merely written down.
#
# Run after any change to the Access application, the tunnel, or the Caddyfile.
#
set -uo pipefail
HOST="${HOST:-router.sainag.com}"
fails=0
ck(){ if [ "$2" = "0" ]; then printf "  [ ok ] %s%s\n" "$1" "${3:+: $3}"
      else printf "  [FAIL] %s%s\n" "$1" "${3:+: $3}"; fails=$((fails+1)); fi; }

echo "Checking https://${HOST} as an anonymous visitor"; echo

code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "https://${HOST}/admin" 2>/dev/null)
[ -n "$code" ] && [ "$code" != "000" ]; ck "endpoint answers" $? "HTTP ${code:-none}"

body=$(curl -sL --max-time 25 "https://${HOST}/admin" 2>/dev/null)
echo "$body" | grep -qiE 'cloudflareaccess\.com|Sign in with|access\.cloudflare'; ck "Access challenge on /admin" $?

# The tell-tale: the UI's own markup coming back unauthenticated.
if echo "$body" | grep -qiE '<title>\s*smart-router\s*</title>'; then
  ck "admin UI NOT served anonymously" 1 "the UI answered without SSO -- Access is NOT protecting this hostname"
else
  ck "admin UI NOT served anonymously" 0
fi

# The API behind it, not just the page.
acode=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "https://${HOST}/admin/api/config" 2>/dev/null)
[ "$acode" != "200" ]; ck "/admin/api/config gated" $? "HTTP ${acode}"

# THE IMPORTANT ONE. /v1 must not exist on this hostname at all.
for path in /v1/chat/completions /health /stats /route; do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "https://${HOST}${path}" 2>/dev/null)
  [ "$c" = "404" ] || [ "$c" = "302" ]; ck "${path} not published" $? "HTTP ${c}"
done

echo
if [ "$fails" -eq 0 ]; then echo "All checks passed."; exit 0
else echo "${fails} check(s) FAILED -- do not consider this hostname safely published."; exit 1; fi
