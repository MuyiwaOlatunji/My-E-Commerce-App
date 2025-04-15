document.addEventListener('DOMContentLoaded', () => {
    // Handle Add to Cart forms
    const forms = document.querySelectorAll('form[action*="/buy/"]');
    
    forms.forEach(form => {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(form);
            const action = form.getAttribute('action');
            
            try {
                const response = await fetch(action, {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                showToast(data.message, data.success ? 'success' : 'error');
                
                if (data.success) {
                    // Update cart count
                    const cartLink = document.querySelector('.navbar-nav a[href="/cart"]');
                    if (cartLink) {
                        const currentText = cartLink.textContent;
                        const currentCount = currentText.includes('(') 
                            ? parseInt(currentText.match(/\((\d+)\)/)?.[1] || 0)
                            : 0;
                        const newCount = currentCount + parseInt(formData.get('quantity') || 1);
                        cartLink.textContent = `Cart${newCount > 0 ? ` (${newCount})` : ''}`;
                    }
                }
            } catch (error) {
                console.error('Add to cart error:', error);
                showToast('Failed to add to cart', 'error');
            }
        });
    });

    // Handle Checkout form
    const checkoutForm = document.querySelector('form[action="/checkout"]');
    if (checkoutForm) {
        checkoutForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            try {
                const response = await fetch('/checkout', {
                    method: 'POST'
                });
                const data = await response.json();
                
                showToast(data.message, data.success ? 'success' : 'error');
                
                if (data.success) {
                    // Update cart count to 0
                    const cartLink = document.querySelector('.navbar-nav a[href="/cart"]');
                    if (cartLink) {
                        cartLink.textContent = 'Cart';
                    }
                    // Reload page to show empty cart
                    setTimeout(() => {
                        window.location.reload();
                    }, 2000);
                }
            } catch (error) {
                console.error('Checkout error:', error);
                showToast('Checkout failed', 'error');
            }
        });
    }
});

function showToast(message, type) {
    const toast = document.getElementById('toast');
    if (!toast) {
        console.warn('Toast element not found');
        return;
    }
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.style.display = 'block';
    
    setTimeout(() => {
        toast.style.display = 'none';
    }, 3000);
}