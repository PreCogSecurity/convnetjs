# Serves the ConvNetJS demos as a static site. The repository root is the
# document root so that demo/*.html can keep referencing ../build/convnet.js.
FROM nginx:alpine
COPY . /usr/share/nginx/html
EXPOSE 80