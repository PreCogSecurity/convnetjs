# Serves the ConvNetJS demos as a static site. The repository root is the
# document root so that demo/*.html can keep referencing ../build/convnet.js.
#
# Hardening notes:
#  * .dockerignore keeps .git/, node_modules/ and the local coverage output out
#    of the build context and out of the document root. Serving .git/ over HTTP
#    publishes the entire repository history to anyone who can reach the
#    container, which is a real information disclosure, not a theoretical one.
#  * The nginx workers run as a non-root user (the unprivileged `nginx` account
#    shipped in the base image) instead of the default root master process.
#  * A read-only root filesystem is requested in docker-compose.yml; the only
#    writable paths are the nginx cache/temp directories.
FROM nginx:1.27-alpine

# Serve from a dedicated document root rather than a whole-repo copy, so the
# image only ever contains files the demos actually need.
ENV NGINX_DOCROOT=/usr/share/nginx/html

COPY --chown=nginx:nginx build/ ${NGINX_DOCROOT}/build/
COPY --chown=nginx:nginx demo/ ${NGINX_DOCROOT}/demo/
COPY --chown=nginx:nginx LICENSE README.md ${NGINX_DOCROOT}/

USER nginx

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://127.0.0.1:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
