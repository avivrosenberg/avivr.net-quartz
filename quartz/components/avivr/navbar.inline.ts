document.addEventListener("nav", () => {
  const navMenu = document.getElementById("navMenu")
  const menuIcon = document.getElementById("menuIcon")

  if (navMenu === null || menuIcon === null) return

  // Define the toggle function
  function toggleMenu() {
    navMenu.classList.toggle("active")
  }

  // Define the outside click handler
  function handleClickOutside(event) {
    if (!menuIcon.contains(event.target) && !navMenu.contains(event.target)) {
      navMenu.classList.remove("active")
    }
  }

  // Add event listeners
  menuIcon.addEventListener("click", toggleMenu)
  document.addEventListener("click", handleClickOutside)

  // Cleanup function to remove event listeners
  window.addCleanup(() => {
    menuIcon.removeEventListener("click", toggleMenu)
    document.removeEventListener("click", handleClickOutside)
  })
})
