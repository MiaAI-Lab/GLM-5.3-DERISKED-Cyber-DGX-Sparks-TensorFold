#!/usr/bin/env bash
# Load the release pins from the same checker used by the converters and by launch preflight.
_exlr8_settings=$(EXLR8_RELEASE="${EXLR8_RELEASE:-}" EXLR8_REV="${EXLR8_REV:-}" EXLR8_PROFILE="${EXLR8_PROFILE:-}" \
  python3 -IB "$(dirname -- "${BASH_SOURCE[0]}")/dirstate.py" settings) || return 1
read -r EXLR8_RELEASE EXLR8_REV EXLR8_PROFILE EXLR8_MANIFEST EXLR8_INTERIM_LEAF EXLR8_NATIVE_LEAF \
  EXLR8_INDEX_SHA EXLR8_SERVED_NAME EXLR8_REPO <<< "$_exlr8_settings"
unset _exlr8_settings
export EXLR8_RELEASE
