/* ===========================
   BLOG SCRIPT.JS
   =========================== */

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Highlight active TOC link on scroll
function updateActiveTocLink() {
    const sections = document.querySelectorAll('.content-section[id]');
    const tocLinks = document.querySelectorAll('.toc-link');

    let currentSection = '';

    sections.forEach(section => {
        const sectionTop = section.offsetTop - 150;
        const sectionHeight = section.offsetHeight;

        if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
            currentSection = section.getAttribute('id');
        }
    });

    tocLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${currentSection}`) {
            link.classList.add('active');
        }
    });
}

// Throttled scroll handler for performance
let scrollTimeout;
window.addEventListener('scroll', () => {
    if (scrollTimeout) {
        clearTimeout(scrollTimeout);
    }
    scrollTimeout = setTimeout(updateActiveTocLink, 50);
});

// Reading progress indicator (optional)
function createReadingProgress() {
    const progressBar = document.createElement('div');
    progressBar.id = 'reading-progress';
    progressBar.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 0%;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #ec4899);
        z-index: 9999;
        transition: width 0.1s ease;
    `;
    document.body.appendChild(progressBar);

    window.addEventListener('scroll', () => {
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight - windowHeight;
        const scrolled = window.scrollY;
        const progress = (scrolled / documentHeight) * 100;

        progressBar.style.width = `${Math.min(progress, 100)}%`;
    });
}

// Copy code button functionality
document.querySelectorAll('pre code').forEach(codeBlock => {
    const pre = codeBlock.parentElement;

    // Create copy button
    const copyButton = document.createElement('button');
    copyButton.className = 'code-copy-btn';
    copyButton.textContent = 'Copy';
    copyButton.style.cssText = `
        position: absolute;
        top: 0.5rem;
        right: 0.5rem;
        padding: 0.25rem 0.75rem;
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 4px;
        font-size: 0.75rem;
        cursor: pointer;
        opacity: 0;
        transition: opacity 0.2s, background 0.2s;
    `;

    pre.style.position = 'relative';
    pre.appendChild(copyButton);

    // Show button on hover
    pre.addEventListener('mouseenter', () => {
        copyButton.style.opacity = '1';
    });

    pre.addEventListener('mouseleave', () => {
        copyButton.style.opacity = '0';
    });

    // Copy functionality
    copyButton.addEventListener('click', async () => {
        const code = codeBlock.textContent;
        try {
            await navigator.clipboard.writeText(code);
            copyButton.textContent = 'Copied!';
            copyButton.style.background = 'rgba(16, 185, 129, 0.2)';
            copyButton.style.borderColor = 'rgba(16, 185, 129, 0.4)';

            setTimeout(() => {
                copyButton.textContent = 'Copy';
                copyButton.style.background = 'rgba(255, 255, 255, 0.1)';
                copyButton.style.borderColor = 'rgba(255, 255, 255, 0.2)';
            }, 2000);
        } catch (err) {
            console.error('Failed to copy code:', err);
        }
    });
});

// Estimate reading time (already in HTML, but can be dynamic)
function calculateReadingTime() {
    const article = document.querySelector('.article-content');
    if (!article) return 0;

    const text = article.textContent;
    const wordsPerMinute = 200;
    const wordCount = text.trim().split(/\s+/).length;
    const readingTime = Math.ceil(wordCount / wordsPerMinute);

    return readingTime;
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    // Create reading progress bar
    createReadingProgress();

    // Initial TOC highlight
    updateActiveTocLink();

    // Log page load
    console.log('Blog loaded successfully! ☕');
    console.log(`Estimated reading time: ${calculateReadingTime()} minutes`);

    // Add smooth reveal animations to elements
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe elements for animation
    document.querySelectorAll('.feature-card, .challenge-item, .solution-card, .related-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});

// Handle checklist interactions
document.querySelectorAll('.checklist-item input[type="checkbox"]').forEach(checkbox => {
    checkbox.addEventListener('change', function() {
        const item = this.closest('.checklist-item');
        if (this.checked) {
            item.style.opacity = '0.6';
            item.style.background = '#f0fdf4';
        } else {
            item.style.opacity = '1';
            item.style.background = 'white';
        }
    });
});

// Add keyboard navigation for accessibility
document.addEventListener('keydown', (e) => {
    // Navigate TOC with arrow keys when focused
    if (document.activeElement.classList.contains('toc-link')) {
        const tocLinks = Array.from(document.querySelectorAll('.toc-link'));
        const currentIndex = tocLinks.indexOf(document.activeElement);

        if (e.key === 'ArrowDown' && currentIndex < tocLinks.length - 1) {
            e.preventDefault();
            tocLinks[currentIndex + 1].focus();
        } else if (e.key === 'ArrowUp' && currentIndex > 0) {
            e.preventDefault();
            tocLinks[currentIndex - 1].focus();
        }
    }
});
