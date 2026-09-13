# The page is static and detection runs in the visitor's browser, so this image
# is nothing but nginx in front of web/.
#
# The `unprivileged` variant of the official image: it already runs as a
# non-root user on port 8080, so the container needs no capability handed back
# to it — plain `nginx` starts as root and would want CHOWN, SETUID and SETGID
# just to set up its cache directories.
FROM nginxinc/nginx-unprivileged:stable-alpine

# conf.d/default.conf and not nginx.conf: the main config is what puts the pid
# file and the temp directories under /tmp, the only place this user can write.
# Overwriting it would throw away exactly what the variant is for.
COPY nginx.conf /etc/nginx/conf.d/default.conf

# web/ alone — the notebook and docs/ have nothing to do here. The build context
# is the repository root rather than web/ because nginx.conf sits beside it.
COPY web/ /usr/share/nginx/html/

EXPOSE 8080
