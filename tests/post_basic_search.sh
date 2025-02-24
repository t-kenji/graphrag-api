#!/usr/bin/env bash
set -eu

DOMAIN=${DOMAIN-"examlpe"}
QUERY=${QUERY-"test query"}
COMMUNITY_LEVEL=${COMMUNITY_LEVEL-2}

# API のホストとポート（デフォルトは 3080）
API_PORT=${API_PORT-3080}
BASE_URL="http://localhost:${API_PORT}"

# JSONペイロード（POSTリクエスト用）
JSON_PAYLOAD=$(cat << __EOD__
{
  "query": "${QUERY}",
  "community_level": ${COMMUNITY_LEVEL}
}
__EOD__
)

echo "POST /v1/search/${DOMAIN}/basic"
time curl -s -X POST "${BASE_URL}/v1/search/${DOMAIN}/basic" \
          -H "Content-Type: application/json" \
          -d "${JSON_PAYLOAD}" | jq
