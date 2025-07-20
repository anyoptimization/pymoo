// Fix invalid IDs before Bootstrap processes them
(function() {
    function sanitizeIds() {
        // Find all elements with IDs containing invalid CSS selector characters
        const elements = document.querySelectorAll('[id]');
        elements.forEach(element => {
            const id = element.id;
            if (id && /[^a-zA-Z0-9_-]/.test(id)) {
                const newId = id.replace(/[^a-zA-Z0-9_-]/g, '-');
                element.id = newId;
                
                // Update any links that reference the old ID
                const links = document.querySelectorAll(`a[href="#${CSS.escape(id)}"]`);
                links.forEach(link => {
                    link.href = `#${newId}`;
                });
            }
        });
    }
    
    // Run immediately if DOM is ready, otherwise wait
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', sanitizeIds);
    } else {
        sanitizeIds();
    }
})();