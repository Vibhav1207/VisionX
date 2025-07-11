document.addEventListener('DOMContentLoaded', () => {
  const button = document.querySelector('.button');
  button.addEventListener('click', () => {
    button.classList.add('clicked');
    setTimeout(() => {
      button.classList.remove('clicked');
    }, 200);
  });
});
