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
                    const cartLink = document.querySelector('a[href="/cart"]');
                    if (cartLink) {
                        const currentCount = parseInt(cartLink.textContent.match(/\((\d+)\)/)?.[1] || 0);
                        const newCount = currentCount + parseInt(formData.get('quantity'));
                        cartLink.innerHTML = `<i class="fas fa-shopping-cart"></i> Cart ${newCount > 0 ? `(${newCount})` : ''}`;
                    }
                }
            } catch (error) {
                showToast('An error occurred', 'error');
            }
        });
    });

    // Handle Checkout button
    const checkoutBtn = document.getElementById('checkout-btn');
    if (checkoutBtn) {
        checkoutBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/checkout', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                const data = await response.json();
                
                showToast(data.message, data.success ? 'success' : 'error');
                
                if (data.success) {
                    // Update cart count to 0
                    const cartLink = document.querySelector('a[href="/cart"]');
                    if (cartLink) {
                        cartLink.innerHTML = `<i class="fas fa-shopping-cart"></i> Cart`;
                    }
                    // Reload page to show empty cart
                    setTimeout(() => {
                        window.location.reload();
                    }, 2000);
                }
            } catch (error) {
                showToast('An error occurred during checkout', 'error');
            }
        });
    }
});

function showToast(message, type) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.style.display = 'block';
    
    setTimeout(() => {
        toast.style.display = 'none';
    }, 3000);
}