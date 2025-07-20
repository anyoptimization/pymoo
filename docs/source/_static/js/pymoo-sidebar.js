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

// Fix sidebar toggle functionality
document.addEventListener('DOMContentLoaded', function() {
    // Find and fix sidebar toggle buttons
    const toggleButtons = document.querySelectorAll('[data-bs-toggle="offcanvas"], .sidebar-toggle');
    
    toggleButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const sidebar = document.querySelector('.bd-sidebar-primary');
            if (sidebar) {
                sidebar.classList.toggle('show');
                // Also toggle a body class for overlay
                document.body.classList.toggle('sidebar-open');
            }
        });
    });
    
    loadNewsletterModal();
});

// Function to load newsletter modal
function loadNewsletterModal() {
    const modalHTML = `
    <!-- Newsletter Modal -->
    <div class="modal fade" id="newsletter-modal" tabindex="-1" role="dialog" aria-labelledby="newsletterModalLabel" aria-hidden="true">
        <div class="modal-dialog" role="document">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="newsletterModalLabel"><b>Newsletter</b></h5>
                    <button type="button" class="close" onclick="closeNewsletter()" aria-label="Close" style="position: absolute; right: 15px; top: 15px; background: none; border: none; font-size: 1.5rem; cursor: pointer;">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
                <div class="modal-body">
                    <!-- Begin Mailchimp Signup Form -->
                    <div id="mc_embed_signup">
                        <form action="https://pymoo.us1.list-manage.com/subscribe/post?u=4454ea68ad3065ee471d6f9aa&amp;id=6f71107645"
                              method="post" id="mc-embedded-subscribe-form" name="mc-embedded-subscribe-form"
                              class="validate" target="_blank" novalidate>
                            <div id="mc_embed_signup_scroll">
                                <div class="indicates-required"><span class="asterisk">*</span> indicates required</div>
                                <div class="mc-field-group">
                                    <label for="mce-EMAIL">Email Address <span class="asterisk">*</span></label>
                                    <input type="email" value="" name="EMAIL" class="required email form-control" id="mce-EMAIL" autocomplete="email">
                                </div>
                                <div class="mc-field-group">
                                    <label for="mce-FNAME">First Name</label>
                                    <input type="text" value="" name="FNAME" class="form-control" id="mce-FNAME">
                                </div>
                                <div class="mc-field-group">
                                    <label for="mce-LNAME">Last Name</label>
                                    <input type="text" value="" name="LNAME" class="form-control" id="mce-LNAME">
                                </div>
                                <div class="mc-field-group">
                                    <label for="mce-INSTITUTIO">Institution</label>
                                    <input type="text" value="" name="INSTITUTIO" class="form-control" id="mce-INSTITUTIO">
                                </div>
                                <p class="mt-3"><small>Powered by <a href="http://eepurl.com/hsCRrP" title="MailChimp - email marketing made easy and fun">MailChimp</a></small></p>
                                <div id="mce-responses" class="clear">
                                    <div class="response" id="mce-error-response" style="display:none"></div>
                                    <div class="response" id="mce-success-response" style="display:none"></div>
                                </div>
                                <!-- real people should not fill this in and expect good things - do not remove this or risk form bot signups-->
                                <div style="position: absolute; left: -5000px;" aria-hidden="true">
                                    <input type="text" name="b_4454ea68ad3065ee471d6f9aa_6f71107645" tabindex="-1" value="">
                                </div>
                                <div class="clear mt-3">
                                    <input type="submit" value="Subscribe" name="subscribe" id="mc-embedded-subscribe" class="btn btn-primary">
                                </div>
                            </div>
                        </form>
                    </div>
                    <!--End mc_embed_signup-->
                </div>
            </div>
        </div>
    </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
}

// Function to handle newsletter modal
function openNewsletter() {
    console.log('Opening newsletter modal...');
    const modal = document.getElementById('newsletter-modal');
    console.log('Modal element:', modal);
    
    if (modal) {
        // Simple show/hide approach
        modal.classList.add('show');
        console.log('Modal should be visible now');
    } else {
        console.log('Modal not found, using fallback');
        // Fallback to direct link
        window.open('https://pymoo.us1.list-manage.com/subscribe?u=4454ea68ad3065ee471d6f9aa&id=6f71107645', '_blank');
    }
}

// Function to close newsletter modal
function closeNewsletter() {
    console.log('Closing newsletter modal...');
    const modal = document.getElementById('newsletter-modal');
    if (modal) {
        modal.style.display = 'none';
        modal.classList.remove('show');
        console.log('Modal closed');
    }
}

// Close modal when clicking outside of it
document.addEventListener('click', function(event) {
    const modal = document.getElementById('newsletter-modal');
    if (modal && event.target === modal) {
        closeNewsletter();
    }
});