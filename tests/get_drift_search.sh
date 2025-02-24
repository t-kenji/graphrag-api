#!/usr/bin/env bash
set -eu

DOMAIN=${DOMAIN-"examlpe"}
QUERY=${QUERY-"test query"}
COMMUNITY_LEVEL=${COMMUNITY_LEVEL-2}

# API のホストとポート（デフォルトは 3080）
API_PORT=${API_PORT-3080}
BASE_URL="http://localhost:${API_PORT}"

echo "GET /v1/search/${DOMAIN}/drift"
time curl -s -G "${BASE_URL}/v1/search/${DOMAIN}/drift" \
          --data-urlencode "query=${QUERY}" \
          --data-urlencode "community_level=${COMMUNITY_LEVEL}" | jq
