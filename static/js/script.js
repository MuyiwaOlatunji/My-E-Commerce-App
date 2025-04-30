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
            const formData = new FormData(checkoutForm);
            try {
                const response = await fetch('/checkout', {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'Accept': 'application/json'
                    }
                });
                const data = await response.json();
                showToast(data.message, data.success ? 'success' : 'error');
                if (data.success && data.redirect) {
                    window.location.href = data.redirect;
                }
            } catch (error) {
                console.error('Checkout error:', error);
                showToast('Checkout failed', 'error');
            }
        });
    }

    // Poll payment status if payment_id exists
    const paymentIdElement = document.querySelector('p[data-payment-id]');
    if (paymentIdElement) {
        const paymentId = paymentIdElement.dataset.paymentId;
        const statusMessage = document.createElement('p');
        statusMessage.id = 'payment-status';
        statusMessage.textContent = 'Checking payment status...';
        paymentIdElement.insertAdjacentElement('afterend', statusMessage);

        const checkPaymentStatus = async () => {
            try {
                const response = await fetch(`/payment/status/${paymentId}`, {
                    headers: {
                        'Accept': 'application/json'
                    }
                });
                const data = await response.json();
                if (data.status === 'success' && data.redirect) {
                    showToast('Payment confirmed! Redirecting to home page...', 'success');
                    setTimeout(() => {
                        window.location.href = data.redirect;
                    }, 2000); // Delay redirect to show toast
                } else if (data.status === 'error') {
                    statusMessage.textContent = 'Error checking payment status.';
                    showToast('Error checking payment status', 'error');
                } else {
                    statusMessage.textContent = 'Payment is still pending...';
                    setTimeout(checkPaymentStatus, 5000); // Check every 5 seconds
                }
            } catch (error) {
                console.error('Payment status check error:', error);
                statusMessage.textContent = 'Failed to check payment status.';
                showToast('Failed to check payment status', 'error');
            }
        };
        checkPaymentStatus();
    }
});

function validateRegister() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    if (!username || username.length < 3) {
        showToast('Username must be at least 3 characters', 'error');
        return false;
    }
    if (!password || password.length < 8) {
        showToast('Password must be at least 8 characters', 'error');
        return false;
    }
    return true;
}

function showToast(message, type) {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.style.display = 'block';
    setTimeout(() => {
        toast.style.display = 'none';
    }, 3000);
}
let pollingAttempts = 0;
const maxAttempts = 120; // 10 minutes (120 * 5 seconds)
const checkPaymentStatus = async () => {
    if (pollingAttempts >= maxAttempts) {
        statusMessage.textContent = 'Payment not confirmed within 10 minutes. Please try again or contact support.';
        showToast('Payment timeout', 'error');
        return;
    }
    pollingAttempts++;
    // Rest of the polling logic...
};