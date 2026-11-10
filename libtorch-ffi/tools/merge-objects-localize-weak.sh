#!/bin/sh
# GHC's "merge objects" step (ld -r) for libtorch-ffi on macOS.
#
# The inline-c blocks in many modules instantiate the same libtorch inline
# virtuals, so their merged objects each carry a weak, auto-hidden copy of
# vtables such as c10::intrusive_ptr_target's. GHC's objects have no
# MH_SUBSECTIONS_VIA_SYMBOLS, so the linker cannot split those copies out to
# coalesce them. The classic linker's ld -r turned auto-hidden weak symbols
# into local ones, so they never met; ld-prime (Xcode / Command Line Tools
# 15+, and the only linker from 27 on, where -ld_classic is ignored) keeps
# them weak, and the shared library link fails on "duplicate symbol".
#
# This restores the classic result: run ld -r, then make every weak
# auto-hidden symbol in the output local. Such a symbol is by definition not
# exported from the image, so localizing it changes nothing a caller can see.
set -e

# -r is ours to pass. GHC's settings file supplies it as "Merge objects
# flags", but -pgmlm replaces the program together with those flags, so the
# arguments that arrive here are only -o and the object list. Without it ld
# tries to link an executable and fails on _main and every RTS symbol.
/usr/bin/ld -r "$@"

# Find -o, also inside the response files GHC passes as @file (one argument
# per line, wrapped in double quotes).
out=""
want_out=0
scan() {
  for arg in "$@"; do
    if [ "$want_out" = 1 ]; then out="$arg"; want_out=0; continue; fi
    case "$arg" in
      -o) want_out=1 ;;
      @*) scan_file "${arg#@}" ;;
    esac
  done
}
scan_file() {
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#\"}"; line="${line%\"}"
    if [ "$want_out" = 1 ]; then out="$line"; want_out=0; continue; fi
    if [ "$line" = "-o" ]; then want_out=1; fi
  done < "$1"
}
scan "$@"
if [ -z "$out" ] || [ ! -f "$out" ]; then exit 0; fi

syms=$(mktemp "${TMPDIR:-/tmp}/localize-weak.XXXXXX")
trap 'rm -f "$syms"' EXIT
/usr/bin/nm -m "$out" | awk '/ weak external automatically hidden / { print $NF }' > "$syms"
if [ -s "$syms" ]; then /usr/bin/nmedit -R "$syms" "$out"; fi
