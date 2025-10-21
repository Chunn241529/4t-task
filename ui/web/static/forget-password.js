class AIAssistantForgetPasswordForm {
    constructor() {
        this.form = document.getElementById('forgetPasswordForm');
        this.emailInput = document.getElementById('email');
        this.submitButton = this.form.querySelector('.neural-button');
        this.successMessage = document.getElementById('successMessage');
        this.emailError = document.getElementById('emailError');
        this.API_BASE_URL = 'https://living-tortoise-polite.ngrok-free.app';
        this.init();
    }

    init() {
        console.log('Initializing forget password form, checking DOM elements...');
        console.log('forgetPasswordForm:', this.form);
        console.log('successMessage:', this.successMessage);
        console.log('emailInput:', this.emailInput);

        this.bindEvents();
        this.setupAIEffects();
        this.emailInput.setAttribute('placeholder', ' ');
    }

    bindEvents() {
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        this.emailInput.addEventListener('blur', () => this.validateEmail());
        this.emailInput.addEventListener('input', () => this.clearError('email'));
    }

    setupAIEffects() {
        this.emailInput.addEventListener('focus', (e) => {
            this.triggerNeuralEffect(e.target.closest('.smart-field'));
        });
    }

    triggerNeuralEffect(field) {
        const indicator = field.querySelector('.ai-indicator');
        indicator.style.opacity = '1';
        setTimeout(() => {
            indicator.style.opacity = '';
        }, 2000);
    }

    validateEmail() {
        const email = this.emailInput.value.trim();
        if (!email) {
            this.showError('email', 'Vui lòng nhập email');
            return false;
        }
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
            this.showError('email', 'Email không hợp lệ');
            return false;
        }
        this.clearError('email');
        return true;
    }

    showError(field, message) {
        const smartField = document.getElementById(field).closest('.smart-field');
        const errorElement = document.getElementById(`${field}Error`);
        smartField.classList.add('error');
        errorElement.textContent = message;
        errorElement.classList.add('show');
    }

    clearError(field) {
        const smartField = document.getElementById(field).closest('.smart-field');
        const errorElement = document.getElementById(`${field}Error`);
        smartField.classList.remove('error');
        errorElement.classList.remove('show');
        setTimeout(() => {
            errorElement.textContent = '';
        }, 200);
    }

    setLoading(loading) {
        this.submitButton.classList.toggle('loading', loading);
        this.submitButton.disabled = loading;
    }

    async handleSubmit(e) {
        e.preventDefault();
        const isEmailValid = this.validateEmail();
        if (!isEmailValid) {
            return;
        }

        this.setLoading(true);
        try {
            const response = await fetch(`${this.API_BASE_URL}/auth/forgetpw`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: this.emailInput.value.trim()
                })
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || 'Đã xảy ra lỗi khi gửi link đặt lại');
            }

            this.form.style.transform = 'scale(0.95)';
            this.form.style.opacity = '0';
            setTimeout(() => {
                this.form.style.display = 'none';
                const signupSection = document.querySelector('.signup-section');
                if (signupSection) {
                    signupSection.style.display = 'none';
                    console.log('Hid .signup-section in showNeuralSuccess');
                } else {
                    console.warn('Element with class .signup-section not found');
                }
                const loginHeader = document.querySelector('.login-header');
                if (loginHeader) {
                    loginHeader.style.display = 'none';
                    console.log('Hid .login-header in showNeuralSuccess');
                } else {
                    console.warn('Element with class .login-header not found');
                }
                this.successMessage.style.display = 'block';
                this.successMessage.classList.add('show');
            }, 300);
        } catch (error) {
            console.error('Forget password failed:', error);
            this.showError('email', error.message);
            this.setLoading(false);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const form = new AIAssistantForgetPasswordForm();
});
