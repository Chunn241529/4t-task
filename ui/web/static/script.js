class AIAssistantLoginForm {
    constructor() {
        this.form = document.getElementById('loginForm');
        this.emailInput = document.getElementById('email');
        this.passwordInput = document.getElementById('password');
        this.passwordToggle = document.getElementById('passwordToggle');
        this.submitButton = this.form.querySelector('.neural-button');
        this.successMessage = document.getElementById('successMessage');
        this.tokenWidget = document.getElementById('tokenWidget');
        this.tokenValue = document.getElementById('tokenValue');
        this.socialButtons = document.querySelectorAll('.social-neural');
        this.API_BASE_URL = 'https://living-tortoise-polite.ngrok-free.app'; // Change to your API URL
        this.deviceId = this.generateDeviceId();
        this.userId = null;
        this.authCode = new URLSearchParams(window.location.search).get('code');
        this.state = new URLSearchParams(window.location.search).get('state');
        this.redirectUri = new URLSearchParams(window.location.search).get('redirect_uri');
        this.init();
    }

    init() {
        // Add debugging to check if required elements exist
        console.log('Initializing form, checking DOM elements...');
        console.log('loginForm:', document.getElementById('loginForm'));
        console.log('neural-social:', document.querySelector('.neural-social'));
        console.log('signup-section:', document.querySelector('.signup-section'));
        console.log('auth-separator:', document.querySelector('.auth-separator'));

        this.bindEvents();
        this.setupPasswordToggle();
        this.setupSocialButtons();
        this.setupAIEffects();
        this.emailInput.nextElementSibling.textContent = 'Username or Email';
        if (this.authCode && this.state && this.redirectUri) {
            this.handleOAuthRedirect();
        }
    }

    generateDeviceId() {
        const deviceId = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
        localStorage.setItem('device_id', deviceId);
        return deviceId;
    }

    bindEvents() {
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        this.emailInput.addEventListener('blur', () => this.validateUsernameOrEmail());
        this.passwordInput.addEventListener('blur', () => this.validatePassword());
        this.emailInput.addEventListener('input', () => this.clearError('email'));
        this.passwordInput.addEventListener('input', () => this.clearError('password'));

        this.emailInput.setAttribute('placeholder', ' ');
        this.passwordInput.setAttribute('placeholder', ' ');
    }

    setupPasswordToggle() {
        this.passwordToggle.addEventListener('click', () => {
            const type = this.passwordInput.type === 'password' ? 'text' : 'password';
            this.passwordInput.type = type;
            this.passwordToggle.classList.toggle('toggle-active', type === 'text');
        });
    }

    setupSocialButtons() {
        this.socialButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const provider = button.querySelector('span').textContent.trim();
                this.handleSocialLogin(provider, button);
            });
        });
    }

    setupAIEffects() {
        [this.emailInput, this.passwordInput].forEach(input => {
            input.addEventListener('focus', (e) => {
                this.triggerNeuralEffect(e.target.closest('.smart-field'));
            });
        });
    }

    triggerNeuralEffect(field) {
        const indicator = field.querySelector('.ai-indicator');
        indicator.style.opacity = '1';
        setTimeout(() => {
            indicator.style.opacity = '';
        }, 2000);
    }

    validateUsernameOrEmail() {
        const value = this.emailInput.value.trim();
        if (!value) {
            this.showError('email', 'Username or email required');
            return false;
        }
        this.clearError('email');
        return true;
    }

    validatePassword() {
        const password = this.passwordInput.value;
        if (!password) {
            this.showError('password', 'Security key required for access');
            return false;
        }
        if (password.length < 6) {
            this.showError('password', 'Security key must be at least 6 characters');
            return false;
        }
        this.clearError('password');
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

    async handleSubmit(e) {
        e.preventDefault();
        const isUsernameOrEmailValid = this.validateUsernameOrEmail();
        const isPasswordValid = this.validatePassword();
        if (!isUsernameOrEmailValid || !isPasswordValid) {
            return;
        }
        this.setLoading(true);
        try {
            const response = await fetch(`${this.API_BASE_URL}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    username_or_email: this.emailInput.value.trim(),
                    password: this.passwordInput.value,
                    device_id: this.deviceId
                })
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || 'Neural connection failed');
            }
            if (data.message === 'Verify needed') {
                console.log('Verification needed, showing verification form...');
                this.userId = data.user_id;
                localStorage.setItem('user_id', data.user_id);
                this.showVerificationForm();
            } else if (data.token) {
                localStorage.setItem('user_id', this.userId || '');
                this.showNeuralSuccess(data.token);
            } else {
                throw new Error('Invalid response from server');
            }
        } catch (error) {
            console.error('Login failed:', error);
            this.showError('email', error.message || 'Login failed. Please try again.');
            this.setLoading(false);
        }
    }

    showVerificationForm() {
        console.log('showVerificationForm called');
        if (!this.form) {
            console.error('Error: loginForm not found in DOM');
            return;
        }
        this.form.style.display = 'none';

        // Safely handle DOM queries with detailed logging
        const neuralSocial = document.querySelector('.neural-social');
        const signupSection = document.querySelector('.signup-section');
        const authSeparator = document.querySelector('.auth-separator');

        if (neuralSocial) {
            neuralSocial.style.display = 'none';
            console.log('Hid .neural-social');
        } else {
            console.warn('Element with class .neural-social not found in DOM');
        }
        if (signupSection) {
            signupSection.style.display = 'none';
            console.log('Hid .signup-section');
        } else {
            console.warn('Element with class .signup-section not found in DOM');
        }
        if (authSeparator) {
            authSeparator.style.display = 'none';
            console.log('Hid .auth-separator');
        } else {
            console.warn('Element with class .auth-separator not found in DOM');
        }

        const verificationForm = document.createElement('form');
        verificationForm.className = 'login-form';
        verificationForm.style.display = 'block'; // Ensure the form is visible
        verificationForm.style.opacity = '1'; // Ensure no opacity issues
        verificationForm.innerHTML = `
            <div class="smart-field" data-field="code">
                <div class="field-background"></div>
                <input type="text" id="code" name="code" required placeholder=" ">
                <label for="code">Verification Code</label>
                <div class="ai-indicator">
                    <div class="ai-pulse"></div>
                </div>
                <div class="field-completion"></div>
            </div>
            <span class="error-message" id="codeError"></span>
            <button type="submit" class="neural-button">
                <div class="button-bg"></div>
                <span class="button-text">Verify Connection</span>
                <div class="button-loader">
                    <div class="neural-spinner">
                        <div class="spinner-segment"></div>
                        <div class="spinner-segment"></div>
                        <div class="spinner-segment"></div>
                    </div>
                </div>
                <div class="button-glow"></div>
            </button>
        `;
        console.log('Appending verification form to DOM, parent:', this.form.parentElement);
        if (this.form.parentElement) {
            this.form.parentElement.appendChild(verificationForm);
            console.log('Verification form appended');
        } else {
            console.error('Error: Parent element for loginForm not found');
            return;
        }
        verificationForm.addEventListener('submit', (e) => this.handleVerification(e));
        const codeInput = verificationForm.querySelector('#code');
        if (codeInput) {
            codeInput.focus();
            console.log('Verification form input focused');
        } else {
            console.error('Error: Verification code input not found');
        }
    }

    async handleVerification(e) {
        e.preventDefault();
        const codeInput = e.target.querySelector('#code');
        const code = codeInput.value.trim();
        if (!code) {
            this.showError('code', 'Verification code required');
            return;
        }
        this.setLoading(true, e.target.querySelector('.neural-button'));
        try {
            const response = await fetch(`${this.API_BASE_URL}/auth/verify?user_id=${this.userId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    code: code,
                    device_id: this.deviceId
                })
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || 'Verification failed');
            }
            e.target.remove();
            localStorage.setItem('user_id', this.userId);
            if (this.authCode && this.state && this.redirectUri) {
                await this.completeOAuth(data.token);
            } else {
                this.showNeuralSuccess(data.token);
            }
        } catch (error) {
            console.error('Verification failed:', error);
            this.showError('code', error.message || 'Invalid verification code');
            this.setLoading(false, e.target.querySelector('.neural-button'));
        }
    }

    async handleSocialLogin(provider, button) {
        console.log(`Initializing ${provider} connection...`);
        const originalHTML = button.innerHTML;
        button.style.pointerEvents = 'none';
        button.style.opacity = '0.7';
        const loadingHTML = `
            <div class="social-bg"></div>
            <div style="display: flex; gap: 2px;">
                <div style="width: 3px; height: 12px; background: currentColor; border-radius: 1px; animation: neuralSpinner 1.2s ease-in-out infinite;"></div>
                <div style="width: 3px; height: 12px; background: currentColor; border-radius: 1px; animation: neuralSpinner 1.2s ease-in-out infinite; animation-delay: 0.1s;"></div>
                <div style="width: 3px; height: 12px; background: currentColor; border-radius: 1px; animation: neuralSpinner 1.2s ease-in-out infinite; animation-delay: 0.2s;"></div>
            </div>
            <span>Connecting...</span>
            <div class="social-glow"></div>
        `;
        button.innerHTML = loadingHTML;
        try {
            await new Promise(resolve => setTimeout(resolve, 2000));
            console.log(`Redirecting to ${provider} neural interface...`);
        } catch (error) {
            console.error(`${provider} connection failed: ${error.message}`);
        } finally {
            button.style.pointerEvents = 'auto';
            button.style.opacity = '1';
            button.innerHTML = originalHTML;
        }
    }

    setLoading(loading, button = this.submitButton) {
        button.classList.toggle('loading', loading);
        button.disabled = loading;
        this.socialButtons.forEach(btn => {
            btn.style.pointerEvents = loading ? 'none' : 'auto';
            btn.style.opacity = loading ? '0.5' : '1';
        });
    }

    showNeuralSuccess(token) {
        this.form.style.transform = 'scale(0.95)';
        this.form.style.opacity = '0';
        setTimeout(() => {
            this.form.style.display = 'none';
            const neuralSocial = document.querySelector('.neural-social');
            const signupSection = document.querySelector('.signup-section');
            const authSeparator = document.querySelector('.auth-separator');
            const loginHeader = document.querySelector('.login-header');
            if (neuralSocial) {
                neuralSocial.style.display = 'none';
                console.log('Hid .neural-social in showNeuralSuccess');
            } else {
                console.warn('Element with class .neural-social not found in showNeuralSuccess');
            }
            if (signupSection) {
                signupSection.style.display = 'none';
                console.log('Hid .signup-section in showNeuralSuccess');
            } else {
                console.warn('Element with class .signup-section not found in showNeuralSuccess');
            }
            if (authSeparator) {
                authSeparator.style.display = 'none';
                console.log('Hid .auth-separator in showNeuralSuccess');
            } else {
                console.warn('Element with class .auth-separator not found in showNeuralSuccess');
            }
            if (loginHeader) {
                loginHeader.style.display = 'none';
                console.log('Hid .login-header in showNeuralSuccess');
            } else {
                console.warn('Element with class .login-header not found in showNeuralSuccess');
            }
            this.successMessage.classList.add('show');
            this.tokenValue.value = token;
            this.tokenWidget.style.display = 'block';
            localStorage.setItem('auth_token', token);
            setTimeout(() => {
                console.log('Neural link established - accessing AI workspace...');
                // window.location.href = '/ai-dashboard'; // Uncomment if redirect is needed
            }, 3000);
        }, 300);
    }

    async completeOAuth(token) {
        this.showNeuralSuccess(token);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const form = new AIAssistantLoginForm();
});
