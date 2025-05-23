// Theme Management
const themeManager = {
    init() {
        this.setInitialTheme();
        this.setupThemeToggle();
    },

    setInitialTheme() {
        const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        const savedTheme = localStorage.getItem('theme');
        const currentTheme = savedTheme ? savedTheme : systemTheme;
        document.documentElement.classList.toggle('dark', currentTheme === 'dark');
    },

    setupThemeToggle() {
        window.changeTheme = () => {
            const html = document.documentElement;
            html.classList.toggle('dark');
            localStorage.setItem('theme', html.classList.contains('dark') ? 'dark' : 'light');
        };
    }
};

// Navigation Management
const navManager = {
    async init() {
        await this.loadNavComponents();
        this.setActiveNavLink();
        this.setupNavigationLinks();
    },

    async loadNavComponents() {
        const navPlaceholder = document.getElementById('nav-placeholder');
        if (!navPlaceholder) return;

        try {
            const response = await fetch('nav.html');
            const html = await response.text();
            navPlaceholder.innerHTML = html;
        } catch (error) {
            console.error('Error loading navigation components:', error);
        }
    },

    setActiveNavLink() {
        const currentPath = window.location.pathname;
        const links = document.querySelectorAll('.nav-links');
        
        links.forEach(link => {
            const linkPath = link.getAttribute('href');
            // Check if the current path matches the link path or if we're on the index page
            if (linkPath === currentPath || 
                (currentPath === '/' && linkPath === '/') || 
                (currentPath.endsWith('index.html') && linkPath === '/') ||
                (currentPath.endsWith('about.html') && linkPath === '/about')) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    },

    setupNavigationLinks() {
        const links = document.querySelectorAll('.nav-links');
        links.forEach(link => {
            link.addEventListener('click', (e) => {
                if (link.getAttribute('href').startsWith('/')) {
                    e.preventDefault();
                    const path = link.getAttribute('href');
                    window.location.href = path === '/' ? 'index.html' : path.substring(1) + '.html';
                }
            });
        });
    }
};

// Chat Management
const chatManager = {
    chatHistory: JSON.parse(localStorage.getItem('chatHistory')) || [],
    isWaitingForResponse: false,
    
    init() {
        this.setupElements();
        this.setupEventListeners();
        this.loadChatHistory();
        this.focusInput();
        this.setupVisibilityChange();
        this.setupGlobalKeyboardHandler();
    },

    setupElements() {
        this.chatMessages = document.getElementById('chat-messages');
        this.messageForm = document.getElementById('message-form');
        this.messageInput = document.getElementById('message-input');
        this.clearHistoryBtn = document.getElementById('clear-history');
        this.sendButton = this.messageForm.querySelector('button[type="submit"]');
    },

    setupEventListeners() {
        if (this.messageForm) {
            this.messageForm.addEventListener('submit', (e) => this.handleMessageSubmit(e));
        }
        if (this.clearHistoryBtn) {
            this.clearHistoryBtn.addEventListener('click', () => this.clearHistory());
        }
        if (this.messageInput) {
            this.messageInput.addEventListener('input', () => this.updateSendButtonState());
            // Focus input when clicking anywhere in the chat container
            this.chatMessages.parentElement.addEventListener('click', () => this.focusInput());
        }
    },

    setupGlobalKeyboardHandler() {
        document.addEventListener('keydown', (e) => {
            // Don't capture if input is disabled or if user is typing in input
            if (this.isWaitingForResponse || e.target === this.messageInput) return;

            // Don't capture modifier keys or special keys
            if (e.ctrlKey || e.altKey || e.metaKey || e.key.length > 1) return;

            // Don't capture if user is typing in a form element
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            // Focus input and append the typed character
            this.focusInput();
            const currentValue = this.messageInput.value;
            this.messageInput.value = currentValue + e.key;
            this.updateSendButtonState();
            e.preventDefault();
        });
    },

    setupVisibilityChange() {
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') {
                this.focusInput();
            }
        });
    },

    focusInput() {
        if (this.messageInput && !this.isWaitingForResponse) {
            this.messageInput.focus();
        }
    },

    updateSendButtonState() {
        const hasText = this.messageInput.value.trim().length > 0;
        this.sendButton.disabled = !hasText || this.isWaitingForResponse;
    },

    setWaitingState(isWaiting) {
        this.isWaitingForResponse = isWaiting;
        this.messageInput.disabled = isWaiting;
        this.updateSendButtonState();
        if (!isWaiting) {
            this.focusInput();
        }
    },

    clearHistory() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            this.chatHistory = [];
            this.saveChatHistory();
            this.chatMessages.innerHTML = '';
            this.focusInput();
        }
    },

    addMessage(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `flex ${isUser ? 'justify-end' : 'justify-start'} message-enter`;
        
        const messageBubble = document.createElement('div');
        messageBubble.className = `max-w-[70%] rounded-2xl p-4 ${
            isUser 
                ? 'bg-blue-500 text-white rounded-br-none shadow-sm' 
                : 'bg-white dark:bg-neutral-900 text-gray-800 dark:text-gray-200 rounded-bl-none shadow-sm border border-neutral-200 dark:border-neutral-800'
        }`;
        
        if (!isUser) {
            const typingIndicator = document.createElement('div');
            typingIndicator.className = 'typing-indicator';
            typingIndicator.innerHTML = `
                <span></span>
                <span></span>
                <span></span>
            `;
            messageBubble.appendChild(typingIndicator);
            this.chatMessages.appendChild(messageDiv);
            
            setTimeout(() => {
                messageBubble.innerHTML = message;
                messageDiv.classList.add('message-enter-active');
                this.setWaitingState(false);
            }, 1000);
        } else {
            messageBubble.textContent = message;
            messageDiv.classList.add('message-enter-active');
        }
        
        messageDiv.appendChild(messageBubble);
        this.chatMessages.appendChild(messageDiv);
        
        this.chatMessages.scrollTo({
            top: this.chatMessages.scrollHeight,
            behavior: 'smooth'
        });
    },

    getAIResponse(message) {
        message = message.toLowerCase();
        
        if (message.includes('hello') || message.includes('hi')) {
            return "Hello! I'm here to help. How are you feeling today?";
        } else if (message.includes('sad') || message.includes('depressed')) {
            return "I'm sorry to hear that you're feeling down. Would you like to talk about what's bothering you?";
        } else if (message.includes('happy') || message.includes('good')) {
            return "I'm glad to hear you're feeling positive! What's making you feel this way?";
        } else if (message.includes('anxious') || message.includes('worried')) {
            return "Anxiety can be challenging. Would you like to discuss what's causing your anxiety?";
        } else if (message.includes('thank')) {
            return "You're welcome! I'm here whenever you need to talk.";
        } else {
            return "I understand. Could you tell me more about how you're feeling?";
        }
    },

    saveChatHistory() {
        localStorage.setItem('chatHistory', JSON.stringify(this.chatHistory));
    },

    loadChatHistory() {
        if (this.chatMessages) {
            this.chatHistory.forEach((msg, index) => {
                setTimeout(() => {
                    this.addMessage(msg.message, msg.isUser);
                }, index * 100);
            });
        }
    },

    async handleMessageSubmit(e) {
        e.preventDefault();
        
        const message = this.messageInput.value.trim();
        if (!message) return;
        
        this.setWaitingState(true);
        this.addMessage(message, true);
        this.chatHistory.push({ message, isUser: true });
        
        try {
            // Get main response
            const response = await fetch(`${config.api.url}/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: message
                })
            });

            if (!response.ok) {
                throw new Error('Failed to get response');
            }

            const result = await response.json();
            
            // Get sentiment and mental health classification
            const [sentimentRes, mentalHealthRes] = await Promise.all([
                fetch(`${config.api.url}/sentiment`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: message })
                }),
                fetch(`${config.api.url}/mental-health`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: message })
                })
            ]);

            const sentimentData = await sentimentRes.json();
            const mentalHealthData = await mentalHealthRes.json();
            
            // Add the response with metadata
            this.addMessage(result.response, false);
            this.chatHistory.push({ 
                message: result.response, 
                isUser: false,
                metadata: {
                    sentiment: sentimentData.sentiment,
                    mentalHealth: mentalHealthData.classification
                }
            });
            
            // Save chat history
            this.saveChatHistory();
        } catch (error) {
            console.error('Error getting AI response:', error);
            this.addMessage("I apologize, but I'm having trouble processing your message right now. Please try again later.", false);
        } finally {
            this.setWaitingState(false);
            this.messageInput.value = '';
        }
    }
};

// Initialize everything when the DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    themeManager.init();
    await navManager.init();
    chatManager.init();
});

// Add styles for typing indicator
const style = document.createElement('style');
style.textContent = `
    .typing-indicator {
        display: flex;
        gap: 4px;
        padding: 4px 0;
    }
    .typing-indicator span {
        width: 8px;
        height: 8px;
        background: #e2e8f0;
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out;
    }
    .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
    .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    .dark .typing-indicator span {
        background: #4b5563;
    }
`;
document.head.appendChild(style); 