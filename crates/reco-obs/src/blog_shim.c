/* Tiny C shim around OBS's variadic `blog(int, const char *fmt, ...)`
 * function. Rust's stable ABI doesn't let us call C variadics directly,
 * so we pre-format the message on the Rust side and pass a plain
 * NUL-terminated string through a fixed-arity entry point here.
 *
 * Link order: libobs provides `blog` at runtime (OBS loads libobs.so
 * before any plugin), so we just need to ensure this translation unit
 * resolves against OBS's symbol table.
 *
 * Log-level constants come from <util/base.h>:
 *   LOG_ERROR   = 100
 *   LOG_WARNING = 200
 *   LOG_INFO    = 300
 *   LOG_DEBUG   = 400
 * We pass the level as an int so the Rust side doesn't need to know
 * the exact numeric values (it uses symbolic constants from the
 * bindgen-generated `ffi` module).
 */
#include <obs/util/base.h>

void reco_obs_blog(int level, const char *message)
{
    /* Use "%s" as the format string so any '%' in the message is
     * treated literally. Misquoting here would be a classic format-
     * string vulnerability, so be explicit. */
    blog(level, "%s", message);
}
