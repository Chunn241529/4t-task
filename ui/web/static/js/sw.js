// Service Worker để cache và xử lý các request
const CACHE_NAME = 'static-cache-v1';
const urlsToCache = [
  '../css/main.css',
  '../css/components/head.css',
  '../css/components/suggestions.css',
  '../css/components/chat_container.css',
  '../css/components/prompt_container.css',
  '../css/components/image_preview.css',
  '../css/components/dropdown.css',
  '../css/components/iframe.css',
  '../css/components/search.css',
  '../css/components/responsive_chat.css',
  '../css/components/loading_bars.css',
  '../css/chat.css',
  '../js/chat_module.js',
  '../js/login.js',
];

// Install Service Worker
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        return cache.addAll(urlsToCache);
      })
  );
});

// Fetch resources
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then((response) => {
        if (response) {
          return response; // Trả về response từ cache nếu có
        }
        return fetch(event.request); // Nếu không có trong cache, fetch từ network
      })
  );
});
