class AIAssistantResetPasswordForm {
    constructor() {
        this.form = document.getElementById('resetPasswordForm');
        this.newPasswordInput = document.getElementById('newPassword');
        this.confirmPasswordInput = document.getElementById('confirmPassword');
        this.newPasswordToggle = document.getElementById('newPasswordToggle');
        this.confirmPasswordToggle = document.getElementById('confirmPasswordToggle');
        this.submitButton = this.form.querySelector('.neural-button');
        this.successMessage = document.getElementById('successMessage');
        this.newPasswordError = document.getElementById('newPasswordError');
        this.confirmPasswordError = document.getElementById('confirmPasswordError');
        this.resetTokenInput = document.getElementById('resetToken');
        this.API_BASE_URL = 'https://living-tortoise-polite.ngrok-free.app';
        this.init();
    }

    init() {
        console.log('Initializing reset password form, checking DOM elements...');
        console.log('resetPasswordForm:', this.form);
        console.log('successMessage:', this.successMessage);
        console.log('newPasswordInput:', this.newPasswordInput);
        console.log('confirmPasswordInput:', this.confirmPasswordInput);

        // Lấy reset_token từ URL
        const urlParams = new URLSearchParams(window.location.search);
        const resetToken = urlParams.get('token');
        if (resetToken) {
            this.resetTokenInput.value = resetToken;
            console.log('Reset token found:', resetToken);
        } else {
            this.showError('confirmPassword', 'Không tìm thấy token. Vui lòng sử dụng link trong email.');
            this.submitButton.disabled = true;
            console.error('No reset token found in URL');
            return;
        }

        this.bindEvents();
        this.setupPasswordToggle();
        this.setupAIEffects();
        this.newPasswordInput.setAttribute('placeholder', ' ');
        this.confirmPasswordInput.setAttribute('placeholder', ' ');
    }

    bindEvents() {
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        this.newPasswordInput.addEventListener('blur', () => this.validateNewPassword());
        this.confirmPasswordInput.addEventListener('blur', () => this.validateConfirmPassword());
        this.newPasswordInput.addEventListener('input', () => this.clearError('newPassword'));
        this.confirmPasswordInput.addEventListener('input', () => this.clearError('confirmPassword'));
    }

    setupPasswordToggle() {
        [this.newPasswordToggle, this.confirmPasswordToggle].forEach(toggle => {
            toggle.addEventListener('click', () => {
                const input = toggle.previousElementSibling;
                const type = input.type === 'password' ? 'text' : 'password';
                input.type = type;
                toggle.classList.toggle('toggle-active', type === 'text');
            });
        });
    }

    setupAIEffects() {
        [this.newPasswordInput, this.confirmPasswordInput].forEach(input => {
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

    validateNewPassword() {
        const password = this.newPasswordInput.value.trim();
        if (!password) {
            this.showError('newPassword', 'Vui lòng nhập mật khẩu mới');
            return false;
        }
        if (password.length < 8) {
            this.showError('newPassword', 'Mật khẩu phải có ít nhất 8 ký tự');
            return false;
        }
        this.clearError('newPassword');
        return true;
    }

    validateConfirmPassword() {
        const confirmPassword = this.confirmPasswordInput.value.trim();
        const newPassword = this.newPasswordInput.value.trim();
        if (!confirmPassword) {
            this.showError('confirmPassword', 'Vui lòng xác nhận mật khẩu');
            return false;
        }
        if (confirmPassword !== newPassword) {
            this.showError('confirmPassword', 'Mật khẩu xác nhận không khớp');
            return false;
        }
        this.clearError('confirmPassword');
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
        const isNewPasswordValid = this.validateNewPassword();
        const isConfirmPasswordValid = this.validateConfirmPassword();
        if (!isNewPasswordValid || !isConfirmPasswordValid) {
            return;
        }

        this.setLoading(true);
        try {
            const response = await fetch(`${this.API_BASE_URL}/auth/reset-password`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    reset_token: this.resetTokenInput.value,
                    new_password: this.newPasswordInput.value.trim()
                })
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || 'Đã xảy ra lỗi khi đặt lại mật khẩu');
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
                setTimeout(() => {
                    console.log('Password reset successful, redirecting to login...');
                    window.location.href = '/';
                }, 2000);
            }, 300);
        } catch (error) {
            console.error('Reset password failed:', error);
            this.showError('confirmPassword', error.message);
            this.setLoading(false);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const form = new AIAssistantResetPasswordForm();
});
