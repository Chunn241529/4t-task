class AIRegisterForm {
    constructor() {
        this.form = document.getElementById('registerForm');
        this.usernameInput = document.getElementById('username');
        this.emailInput = document.getElementById('email');
        this.passwordInput = document.getElementById('password');
        this.confirmPasswordInput = document.getElementById('confirmPassword');
        this.genderInput = document.getElementById('gender');
        this.genderToggle = document.getElementById('genderToggle');
        this.genderMenu = document.getElementById('genderMenu');
        this.genderDisplay = document.getElementById('genderDisplay');
        this.passwordToggle = document.getElementById('passwordToggle');
        this.confirmPasswordToggle = document.getElementById('confirmPasswordToggle');
        this.submitButton = this.form.querySelector('.neural-button');
        this.successMessage = document.getElementById('successMessage');
        this.tokenWidget = document.getElementById('tokenWidget');
        this.tokenValue = document.getElementById('tokenValue');
        this.copyTokenButton = document.getElementById('copyToken');
        this.API_BASE_URL = 'https://living-tortoise-polite.ngrok-free.app';
        this.userId = null;
        this.init();
    }

    init() {
        console.log('Initializing register form...');
        console.log('form:', this.form);
        console.log('successMessage:', this.successMessage);
        this.bindEvents();
        this.setupPasswordToggles();
        this.setupAIEffects();
        this.setupCopyToken();
        this.usernameInput.nextElementSibling.textContent = 'Username';
        this.emailInput.nextElementSibling.textContent = 'Email';
        this.passwordInput.nextElementSibling.textContent = 'Mật khẩu';
        this.confirmPasswordInput.nextElementSibling.textContent = 'Xác nhận mật khẩu';
        document.querySelector('.gender-label').textContent = 'Giới tính';
    }

    bindEvents() {
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        this.usernameInput.addEventListener('blur', () => this.validateUsername());
        this.emailInput.addEventListener('blur', () => this.validateEmail());
        this.passwordInput.addEventListener('blur', () => this.validatePassword());
        this.confirmPasswordInput.addEventListener('blur', () => this.validateConfirmPassword());
        this.genderToggle.addEventListener('click', () => {
            this.genderMenu.style.display = this.genderMenu.style.display === 'block' ? 'none' : 'block';
            this.triggerNeuralEffect(this.genderToggle.closest('.smart-field'));
        });
        this.genderMenu.addEventListener('click', (e) => {
            const item = e.target.closest('.dropdown-item');
            if (item) {
                const value = item.getAttribute('data-value');
                const displayText = item.textContent;
                this.genderInput.value = value;
                this.genderDisplay.textContent = displayText;
                this.genderMenu.style.display = 'none';
                this.clearError('gender');
            }
        });
        document.addEventListener('click', (e) => {
            if (!this.genderToggle.contains(e.target) && !this.genderMenu.contains(e.target)) {
                this.genderMenu.style.display = 'none';
            }
        });

        [this.usernameInput, this.emailInput, this.passwordInput, this.confirmPasswordInput].forEach(input => {
            input.setAttribute('placeholder', ' ');
        });
    }

    setupPasswordToggles() {
        this.passwordToggle.addEventListener('click', () => {
            const type = this.passwordInput.type === 'password' ? 'text' : 'password';
            this.passwordInput.type = type;
            this.passwordToggle.classList.toggle('toggle-active', type === 'text');
        });
        this.confirmPasswordToggle.addEventListener('click', () => {
            const type = this.confirmPasswordInput.type === 'password' ? 'text' : 'password';
            this.confirmPasswordInput.type = type;
            this.confirmPasswordToggle.classList.toggle('toggle-active', type === 'text');
        });
    }

    setupAIEffects() {
        [this.usernameInput, this.emailInput, this.passwordInput, this.confirmPasswordInput, this.genderToggle].forEach(element => {
            element.addEventListener('focus', (e) => {
                this.triggerNeuralEffect(e.target.closest('.smart-field'));
            });
        });
    }

    setupCopyToken() {
        this.copyTokenButton.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(this.tokenValue.value);
                this.copyTokenButton.textContent = 'Copied!';
                setTimeout(() => {
                    this.copyTokenButton.textContent = 'Copy Token';
                }, 2000);
            } catch (error) {
                console.error('Failed to copy token:', error);
                this.showError('token', 'Failed to copy token');
            }
        });
    }

    triggerNeuralEffect(field) {
        const indicator = field.querySelector('.ai-indicator');
        indicator.style.opacity = '1';
        setTimeout(() => {
            indicator.style.opacity = '';
        }, 2000);
    }

    validateUsername() {
        const value = this.usernameInput.value.trim();
        console.log('Validating username:', value);
        if (!value) {
            this.showError('username', 'Username is required');
            return false;
        }
        if (value.length < 3) {
            this.showError('username', 'Username must be at least 3 characters');
            return false;
        }
        this.clearError('username');
        return true;
    }

    validateEmail() {
        const value = this.emailInput.value.trim();
        console.log('Validating email:', value);
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!value) {
            this.showError('email', 'Email is required');
            return false;
        }
        if (!emailRegex.test(value)) {
            this.showError('email', 'Invalid email format');
            return false;
        }
        this.clearError('email');
        return true;
    }

    validatePassword() {
        const password = this.passwordInput.value;
        console.log('Validating password:', password.length, 'characters');
        if (!password) {
            this.showError('password', 'Password is required');
            return false;
        }
        if (password.length < 6) {
            this.showError('password', 'Password must be at least 6 characters');
            return false;
        }
        this.clearError('password');
        return true;
    }

    validateConfirmPassword() {
        const password = this.passwordInput.value;
        const confirmPassword = this.confirmPasswordInput.value;
        console.log('Validating confirm password:', confirmPassword.length, 'characters');
        if (!confirmPassword) {
            this.showError('confirmPassword', 'Please confirm your password');
            return false;
        }
        if (confirmPassword !== password) {
            this.showError('confirmPassword', 'Passwords do not match');
            return false;
        }
        this.clearError('confirmPassword');
        return true;
    }

    validateGender() {
        const value = this.genderInput.value;
        console.log('Validating gender:', value);
        this.clearError('gender');
        return true;
    }

    showError(field, message) {
        const smartField = document.getElementById(field)?.closest('.smart-field') || document.querySelector(`[data-field="${field}"]`);
        const errorElement = document.getElementById(`${field}Error`);
        if (smartField && errorElement) {
            smartField.classList.add('error');
            errorElement.textContent = message;
            errorElement.classList.add('show');
        } else {
            console.error(`Error: Could not find elements for field ${field}`);
        }
    }

    clearError(field) {
        const smartField = document.getElementById(field)?.closest('.smart-field') || document.querySelector(`[data-field="${field}"]`);
        const errorElement = document.getElementById(`${field}Error`);
        if (smartField && errorElement) {
            smartField.classList.remove('error');
            errorElement.classList.remove('show');
            setTimeout(() => {
                errorElement.textContent = '';
            }, 200);
        }
    }

    async handleSubmit(e) {
        e.preventDefault();
        console.log('Handling register submit...');
        const isUsernameValid = this.validateUsername();
        const isEmailValid = this.validateEmail();
        const isPasswordValid = this.validatePassword();
        const isConfirmPasswordValid = this.validateConfirmPassword();
        const isGenderValid = this.validateGender();
        console.log('Validation results:', { isUsernameValid, isEmailValid, isPasswordValid, isConfirmPasswordValid, isGenderValid });

        if (!isUsernameValid || !isEmailValid || !isPasswordValid || !isConfirmPasswordValid || !isGenderValid) {
            console.log('Validation failed, stopping submission');
            return;
        }

        this.setLoading(true);

        try {
            const gender = this.genderInput.value || null;
            console.log('Sending request to /register with data:', {
                username: this.usernameInput.value.trim(),
                email: this.emailInput.value.trim(),
                password: this.passwordInput.value,
                gender: gender
                // ĐÃ XÓA: device_id - backend sẽ tự detect
            });

            const response = await fetch(`${this.API_BASE_URL}/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    username: this.usernameInput.value.trim(),
                    email: this.emailInput.value.trim(),
                    password: this.passwordInput.value,
                    gender: gender
                })
            });

            const data = await response.json();
            console.log('API /register response:', data);

            if (!response.ok) {
                throw new Error(data.detail || 'Registration failed');
            }

            this.userId = data.user_id;
            if (!this.userId) {
                console.error('Error: No user_id in response');
                throw new Error('Registration succeeded but no user_id provided');
            }

            localStorage.setItem('user_id', this.userId);
            console.log('Registration successful, showing verification form with user_id:', this.userId);
            this.showVerificationForm();

        } catch (error) {
            console.error('Registration failed:', error);
            this.showError('email', error.message || 'Registration failed. Please try again.');
        } finally {
            this.setLoading(false);
        }
    }

    showVerificationForm() {
        console.log('Showing verification form...');
        this.form.style.display = 'none';
        const signupSection = document.querySelector('.signup-section');
        if (signupSection) {
            signupSection.style.display = 'none';
            console.log('Hid .signup-section');
        }

        const verificationForm = document.createElement('form');
        verificationForm.className = 'login-form';
        verificationForm.id = 'verifyForm';
        verificationForm.style.display = 'block';
        verificationForm.style.opacity = '1';
        verificationForm.innerHTML = `
            <div class="smart-field" data-field="code">
                <div class="field-background"></div>
                <input type="text" id="code" name="code" required placeholder=" " autocomplete="one-time-code">
                <label for="code">Verification Code</label>
                <div class="ai-indicator">
                    <div class="ai-pulse"></div>
                </div>
                <div class="field-completion"></div>
            </div>
            <span class="error-message" id="codeError"></span>
            <br>
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

        this.form.parentElement.appendChild(verificationForm);
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
        console.log('Handling verify submit...');
        const codeInput = e.target.querySelector('#code');
        const code = codeInput.value.trim();
        
        if (!code) {
            this.showError('code', 'Verification code required');
            return;
        }

        if (!this.userId) {
            console.error('Error: No user_id available for verification');
            this.showError('code', 'User ID missing. Please try registering again.');
            return;
        }

        this.setLoading(true, e.target.querySelector('.neural-button'));

        try {
            const response = await fetch(`${this.API_BASE_URL}/auth/verify?user_id=${this.userId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    code: code
                    // ĐÃ XÓA: device_id - backend sẽ tự detect
                }),
                credentials: 'include' // QUAN TRỌNG: Để nhận cookie từ backend
            });

            const data = await response.json();
            console.log('Verification response:', data);

            if (!response.ok) {
                throw new Error(data.detail || 'Verification failed');
            }

            // Xóa form verification
            e.target.remove();
            
            // Lưu thông tin user
            localStorage.setItem('user_id', this.userId);
            localStorage.setItem('auth_token', data.token);
            
            // Hiển thị thành công và chuyển hướng
            this.showSuccess(data.token);

        } catch (error) {
            console.error('Verification failed:', error);
            this.showError('code', error.message || 'Invalid verification code');
            this.setLoading(false, e.target.querySelector('.neural-button'));
        }
    }

    setLoading(loading, button = this.submitButton) {
        if (button) {
            button.classList.toggle('loading', loading);
            button.disabled = loading;
        }
    }

    showSuccess(token) {
        console.log('Showing success message with token:', token);
        
        // Ẩn tất cả các phần không cần thiết
        const signupSection = document.querySelector('.signup-section');
        const loginHeader = document.querySelector('.login-header');
        
        if (signupSection) signupSection.style.display = 'none';
        if (loginHeader) loginHeader.style.display = 'none';

        // Hiển thị success message
        this.successMessage.classList.add('show');
        this.tokenWidget.style.display = 'block';
        this.tokenValue.value = token;
        
        localStorage.setItem('auth_token', token);
        
        console.log('Registration and verification completed - redirecting to chat...');
        
        // Tự động chuyển hướng đến chat sau 3 giây
        setTimeout(() => {
            window.location.href = '/chat';
        }, 3000);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const form = new AIRegisterForm();
});
