// Gradient Trade Analysis - Interactive Navigation

document.addEventListener('DOMContentLoaded', function() {
    // Initialize navigation
    initializeNavigation();
    
    // Add smooth scrolling
    addSmoothScrolling();
    
    // Add intersection observer for animations
    addScrollAnimations();
    
    // Add active page highlighting
    highlightActivePage();
});

function initializeNavigation() {
    const navbar = document.querySelector('.navbar');
    
    // Add scroll effect to navbar
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(13, 17, 23, 0.98)';
        } else {
            navbar.style.background = 'rgba(13, 17, 23, 0.95)';
        }
    });
    
    // Mobile menu toggle (if needed later)
    const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener('click', toggleMobileMenu);
    }
}

function addSmoothScrolling() {
    // Smooth scrolling for internal links
    document.querySelectorAll('a[href^="#"]').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                const offsetTop = targetElement.offsetTop - 80; // Account for fixed navbar
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
}

function addScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);
    
    // Observe all sections and cards
    document.querySelectorAll('.section, .card, .metric-card').forEach(el => {
        observer.observe(el);
    });
}

function highlightActivePage() {
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    const navLinks = document.querySelectorAll('.nav-links a');
    
    navLinks.forEach(link => {
        const linkPage = link.getAttribute('href');
        if (linkPage === currentPage || (currentPage === '' && linkPage === 'index.html')) {
            link.classList.add('active');
        }
    });
}

function toggleMobileMenu() {
    const navLinks = document.querySelector('.nav-links');
    navLinks.classList.toggle('mobile-open');
}

// Math rendering support (if MathJax is loaded)
function renderMath() {
    if (window.MathJax) {
        MathJax.typesetPromise().then(() => {
            console.log('Math rendered successfully');
        }).catch((err) => {
            console.log('Math rendering failed:', err);
        });
    }
}

// Interactive elements
function addInteractiveElements() {
    // Add click handlers for expandable sections
    document.querySelectorAll('.expandable-trigger').forEach(trigger => {
        trigger.addEventListener('click', function() {
            const target = document.querySelector(this.dataset.target);
            if (target) {
                target.classList.toggle('expanded');
                this.classList.toggle('expanded');
            }
        });
    });
    
    // Add hover effects for mathematical formulas
    document.querySelectorAll('.formula').forEach(formula => {
        formula.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.05)';
            this.style.transition = 'transform 0.3s ease';
        });
        
        formula.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
        });
    });
}

// Initialize interactive elements when DOM is ready
document.addEventListener('DOMContentLoaded', addInteractiveElements);

// Page transition effects
function initializePageTransitions() {
    // Add loading animation for page changes
    document.querySelectorAll('a[href$=".html"]').forEach(link => {
        link.addEventListener('click', function(e) {
            // Don't prevent default, just add visual feedback
            this.style.opacity = '0.7';
            setTimeout(() => {
                this.style.opacity = '1';
            }, 200);
        });
    });
}

// Call initialization
document.addEventListener('DOMContentLoaded', initializePageTransitions);