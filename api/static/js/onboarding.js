(function () {
  const ready = (callback) => {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", callback, { once: true });
      return;
    }
    callback();
  };

  ready(() => {
    document.querySelectorAll("[data-scroll-target]").forEach((element) => {
      element.addEventListener("click", () => {
        const selector = element.getAttribute("data-scroll-target");
        const target = selector ? document.querySelector(selector) : null;
        if (target) {
          target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      });
    });
  });
})();
