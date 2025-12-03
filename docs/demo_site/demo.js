(function () {
  const randomBool = () => Math.random() > 0.5;

  function toggleLoginButtonId() {
    const button = document.querySelector('#loginButton, #login-btn');
    if (!button) {
      return;
    }
    const nextId = button.id === 'loginButton' ? 'login-btn' : 'loginButton';
    button.id = nextId;
    button.setAttribute('data-dynamic-id', nextId);
  }

  function scheduleButtonMutation() {
    setTimeout(() => {
      toggleLoginButtonId();
      setInterval(toggleLoginButtonId, 1200);
    }, 1000);
  }

  function maybeShowCookieBanner() {
    const banner = document.getElementById('cookie-banner');
    if (!banner) {
      return;
    }
    const dismiss = document.getElementById('dismiss-cookie');
    if (dismiss) {
      dismiss.addEventListener('click', () => {
        banner.classList.remove('show');
      });
    }
    if (randomBool()) {
      banner.classList.add('show');
    }
  }

  function wireLoginForm() {
    const form = document.getElementById('login-form');
    if (!form) {
      return;
    }
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      setTimeout(() => {
        window.location.href = 'dashboard.html';
      }, randomBool() ? 1500 : 250);
    });
  }

  function delayButtonVisibility() {
    const button = document.querySelector('#loginButton, #login-btn');
    if (!button) {
      return;
    }
    button.style.opacity = '0';
    const delay = randomBool() ? 1200 : 200;
    setTimeout(() => {
      button.style.opacity = '1';
    }, delay);
  }

  window.addEventListener('DOMContentLoaded', () => {
    maybeShowCookieBanner();
    wireLoginForm();
    delayButtonVisibility();
    scheduleButtonMutation();
  });
})();
