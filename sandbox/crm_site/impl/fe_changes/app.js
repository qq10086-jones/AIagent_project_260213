const API_BASE = '';

const LANDING_CONTENT = {
  hero: {
    title: 'Bean & Brew',
    subtitle: 'Artisanal coffee crafted with passion. Experience the finest single-origin beans, handcrafted beverages, and a cozy atmosphere that feels like home.',
    ctaText: 'Reserve a Table'
  },
  menu: [
    {
      category: 'Espresso Drinks',
      items: [
        { name: 'Classic Espresso', price: '¥28', description: 'Rich, bold single shot' },
        { name: 'Americano', price: '¥32', description: 'Espresso with hot water' },
        { name: 'Cappuccino', price: '¥36', description: 'Espresso, steamed milk, foam' },
        { name: 'Latte', price: '¥36', description: 'Espresso with steamed milk' },
        { name: 'Mocha', price: '¥38', description: 'Espresso, chocolate, steamed milk' }
      ]
    },
    {
      category: 'Handcrafted Beverages',
      items: [
        { name: 'Cold Brew', price: '¥35', description: 'Steeped 18 hours' },
        { name: 'Matcha Latte', price: '¥40', description: 'Premium Japanese matcha' },
        { name: 'Chai Latte', price: '¥38', description: 'Spiced black tea' },
        { name: 'Honey Lavender Latte', price: '¥42', description: 'Floral and soothing' }
      ]
    },
    {
      category: 'Bakery',
      items: [
        { name: 'Croissant', price: '¥18', description: 'Buttery French classic' },
        { name: 'Blueberry Muffin', price: '¥22', description: 'Fresh baked daily' },
        { name: 'Avocado Toast', price: '¥45', description: 'Sourdough, avocado, poached egg' },
        { name: 'Bagel & Cream Cheese', price: '¥25', description: 'Plain or everything' }
      ]
    }
  ],
  features: [
    {
      icon: 'quality',
      title: 'Premium Beans',
      description: 'Sourced directly from ethical farms, our single-origin beans are carefully roasted to bring out the unique flavors of each region.'
    },
    {
      icon: 'design',
      title: 'Expert Baristas',
      description: 'Our skilled baristas craft each beverage with precision and passion, ensuring the perfect cup every time.'
    },
    {
      icon: 'sustainability',
      title: 'Eco-Friendly',
      description: 'Committed to sustainability, we use biodegradable cups and source local ingredients whenever possible.'
    }
  ],
  businessHours: {
    weekday: '7:00 AM - 9:00 PM',
    weekend: '8:00 AM - 10:00 PM'
  },
  faqs: [
    {
      question: 'Do you offer dairy-free milk options?',
      answer: 'Yes! We provide oat milk, almond milk, and soy milk as alternatives at no extra charge.'
    },
    {
      question: 'Can I make a reservation?',
      answer: 'Absolutely! Click the "Reserve a Table" button to book your spot. For larger groups, please call us directly.'
    },
    {
      question: 'Do you have Wi-Fi?',
      answer: 'Yes, we offer free high-speed Wi-Fi for all our guests. Password is available at the counter.'
    },
    {
      question: 'Are you pet-friendly?',
      answer: 'We love pets! Our outdoor patio welcomes well-behaved dogs. Water bowls are available for your furry friends.'
    }
  ],
  contact: {
    email: 'hello@beanandbrew.com',
    phone: '+86 400-123-4567',
    address: '88 Coffee Lane, Jing\'an District, Shanghai'
  },
  story: {
    title: 'Our Story',
    subtitle: 'A journey of coffee passion',
    content: [
      {
        heading: 'The Beginning',
        paragraph: 'Founded in 2018, Bean & Brew started as a small corner shop with a simple mission: to serve exceptional coffee that brings people together.'
      },
      {
        heading: 'Our Philosophy',
        paragraph: 'We believe great coffee starts with great relationships—with farmers, with our team, and with you. Every cup tells a story of dedication.'
      },
      {
        heading: 'Today',
        paragraph: 'Now a beloved neighborhood spot, we continue to source the finest beans and create a welcoming space for coffee lovers.'
      }
    ],
    timeline: [
      { year: '2018', event: 'First shop opens in Shanghai' },
      { year: '2020', event: 'Launched wholesale bean program' },
      { year: '2022', event: 'Won Best Coffee Shop award' },
      { year: '2024', event: 'Opened third location' }
    ]
  },
  reviews: [
    { id: 1, name: 'Michael T.', rating: 5, comment: 'Best coffee in Shanghai! The latte art is incredible.', date: '2024-01-18' },
    { id: 2, name: 'Lisa W.', rating: 5, comment: 'Love the cozy atmosphere. Perfect spot to work from.', date: '2024-01-15' },
    { id: 3, name: 'David K.', rating: 4, comment: 'Great cold brew and the pastries are always fresh!', date: '2024-01-12' }
  ]
};

const ICONS = {
  quality: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>',
  design: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 1 0 20"/><path d="M12 2a10 10 0 0 0 0 20"/></svg>',
  sustainability: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22c4-4 8-7.5 8-12a8 8 0 1 0-16 0c0 4.5 4 8 8 12z"/><circle cx="12" cy="10" r="3"/></svg>',
  clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>',
  location: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>',
  phone: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg>'
};

async function fetchApi(endpoint) {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`);
    const data = await response.json();
    if (data.success) {
      return data.data;
    }
    throw new Error(data.error || 'API request failed');
  } catch (error) {
    console.error(`API Error for ${endpoint}:`, error);
    return null;
  }
}

function renderHero(hero) {
  const heroSection = document.querySelector('.hero-title');
  const subtitleEl = document.querySelector('.hero-subtitle');
  const ctaEl = document.querySelector('.hero-cta');
  
  if (heroSection) heroSection.textContent = hero.title;
  if (subtitleEl) subtitleEl.textContent = hero.subtitle;
  if (ctaEl) ctaEl.textContent = hero.ctaText;
}

function renderMenu(menu) {
  const container = document.querySelector('.menu-grid');
  if (!container || !menu) return;
  
  container.innerHTML = menu.map(category => `
    <div class="menu-category">
      <h3 class="menu-category-title">${category.category}</h3>
      <div class="menu-items">
        ${category.items.map(item => `
          <div class="menu-item">
            <div class="menu-item-header">
              <span class="menu-item-name">${item.name}</span>
              <span class="menu-item-price">${item.price}</span>
            </div>
            <p class="menu-item-description">${item.description}</p>
          </div>
        `).join('')}
      </div>
    </div>
  `).join('');
}

function renderBusinessHours(hours) {
  const container = document.getElementById('hoursContent');
  if (!container || !hours) return;
  
  container.innerHTML = `
    <div class="hours-grid">
      <div class="hours-item">
        <div class="hours-icon">${ICONS.clock}</div>
        <h4>Weekdays</h4>
        <p>${hours.weekday}</p>
      </div>
      <div class="hours-item">
        <div class="hours-icon">${ICONS.clock}</div>
        <h4>Weekends</h4>
        <p>${hours.weekend}</p>
      </div>
    </div>
  `;
}

function renderFeatures(features) {
  const container = document.querySelector('.features-grid');
  if (!container) return;
  
  container.innerHTML = features.map(feature => `
    <div class="promo-feature-card">
      <div class="promo-feature-icon">${ICONS[feature.icon] || ICONS.quality}</div>
      <h3>${feature.title}</h3>
      <p>${feature.description}</p>
    </div>
  `).join('');
}

function renderStory(story) {
  const container = document.getElementById('storyContent');
  if (!container || !story) return;
  
  let contentHtml = story.content.map(section => `
    <div class="story-section">
      <h3>${section.heading}</h3>
      <p>${section.paragraph}</p>
    </div>
  `).join('');
  
  let timelineHtml = story.timeline.map(item => `
    <div class="timeline-item">
      <span class="timeline-year">${item.year}</span>
      <span class="timeline-event">${item.event}</span>
    </div>
  `).join('');
  
  container.innerHTML = `
    <div class="story-text">
      ${contentHtml}
    </div>
    <div class="story-timeline">
      <h3>Our Journey</h3>
      ${timelineHtml}
    </div>
  `;
  
  const titleEl = document.querySelector('.story-title');
  const subtitleEl = document.querySelector('.story-subtitle');
  if (titleEl) titleEl.textContent = story.title;
  if (subtitleEl) subtitleEl.textContent = story.subtitle;
}

function renderReviews(reviews) {
  const container = document.querySelector('.reviews-grid');
  if (!container || !reviews) return;
  
  container.innerHTML = reviews.map(review => `
    <div class="review-card">
      <div class="review-header">
        <span class="review-name">${review.name}</span>
        <div class="review-rating">${'★'.repeat(review.rating)}${'☆'.repeat(5 - review.rating)}</div>
      </div>
      <p class="review-comment">${review.comment}</p>
      <span class="review-date">${review.date}</span>
    </div>
  `).join('');
}

function renderFaqs(faqs) {
  const container = document.querySelector('.faq-list');
  if (!container || !faqs) return;
  
  container.innerHTML = faqs.map(faq => `
    <div class="faq-item">
      <button class="faq-question" aria-expanded="false">
        <span>${faq.question}</span>
        <svg class="faq-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M6 9l6 6 6-6"/>
        </svg>
      </button>
      <div class="faq-answer">
        <p>${faq.answer}</p>
      </div>
    </div>
  `).join('');
  
  container.querySelectorAll('.faq-question').forEach(btn => {
    btn.addEventListener('click', () => {
      const isExpanded = btn.getAttribute('aria-expanded') === 'true';
      btn.setAttribute('aria-expanded', !isExpanded);
      btn.classList.toggle('active');
      const answer = btn.nextElementSibling;
      answer.style.maxHeight = isExpanded ? null : answer.scrollHeight + 'px';
    });
  });
}

function renderContact(contact) {
  const footerText = document.querySelector('.promo-footer-text');
  if (footerText && contact) {
    footerText.textContent = `Contact: ${contact.email} | ${contact.phone} | ${contact.address}`;
  }
  
  const addressEl = document.getElementById('storeAddress');
  if (addressEl && contact) {
    addressEl.innerHTML = `
      <div class="contact-info-item">
        <div class="contact-icon">${ICONS.location}</div>
        <div>
          <h4>Visit Us</h4>
          <p>${contact.address}</p>
        </div>
      </div>
      <div class="contact-info-item">
        <div class="contact-icon">${ICONS.phone}</div>
        <div>
          <h4>Call Us</h4>
          <p>${contact.phone}</p>
        </div>
      </div>
      <div class="contact-info-item">
        <div class="contact-icon">${ICONS.location}</div>
        <div>
          <h4>Email</h4>
          <p>${contact.email}</p>
        </div>
      </div>
    `;
  }
}

function handleReservationClick() {
  const reservationSection = document.getElementById('reservation');
  if (reservationSection) {
    reservationSection.scrollIntoView({ behavior: 'smooth' });
    showReservationPlaceholder();
  }
}

function showReservationPlaceholder() {
  let placeholder = document.getElementById('reservationPlaceholder');
  if (!placeholder) {
    const section = document.querySelector('.reservation-section');
    if (section) {
      placeholder = document.createElement('div');
      placeholder.id = 'reservationPlaceholder';
      placeholder.className = 'reservation-placeholder';
      placeholder.innerHTML = `
        <div class="placeholder-content">
          <h4>Reservation System Coming Soon</h4>
          <p>Online reservations will be available shortly. For now, please call us at ${LANDING_CONTENT.contact.phone} to book your table.</p>
          <button class="promo-submit-btn" onclick="this.parentElement.parentElement.remove()">Got it</button>
        </div>
      `;
      section.appendChild(placeholder);
    }
  }
}

function handleCtaClick() {
  handleReservationClick();
}

async function handleContactSubmit(event) {
  const form = event.target;
  const feedbackEl = document.getElementById('contactFeedback');
  const submitBtn = form.querySelector('.promo-submit-btn');
  
  const formData = {
    name: form.name?.value?.trim(),
    email: form.email?.value?.trim(),
    message: form.message?.value?.trim()
  };
  
  if (!formData.name || !formData.email || !formData.message) {
    if (feedbackEl) {
      feedbackEl.textContent = 'Please fill in all required fields.';
      feedbackEl.className = 'promo-feedback promo-feedback-error';
    }
    return;
  }
  
  submitBtn.disabled = true;
  submitBtn.textContent = 'Sending...';
  
  try {
    const response = await fetch(`${API_BASE}/api/contact`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData)
    });
    
    const result = await response.json();
    
    if (result.success) {
      if (feedbackEl) {
        feedbackEl.textContent = result.message || 'Thank you for your message!';
        feedbackEl.className = 'promo-feedback promo-feedback-success';
      }
      form.reset();
    } else {
      throw new Error(result.error || 'Submission failed');
    }
  } catch (error) {
    console.error('Contact form error:', error);
    if (feedbackEl) {
      feedbackEl.textContent = 'Failed to send message. Please try again.';
      feedbackEl.className = 'promo-feedback promo-feedback-error';
    }
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Send Message';
  }
}

function initContactForm() {
  const form = document.getElementById('contactForm');
  const formFields = document.querySelector('.form-fields');
  
  if (formFields && !formFields.querySelector('input')) {
    formFields.innerHTML = `
      <div class="promo-form-group">
        <label for="name">Name *</label>
        <input type="text" id="name" name="name" required placeholder="Your name">
      </div>
      <div class="promo-form-group">
        <label for="email">Email *</label>
        <input type="email" id="email" name="email" required placeholder="your@email.com">
      </div>
      <div class="promo-form-group full">
        <label for="message">Message *</label>
        <textarea id="message" name="message" rows="4" required placeholder="How can we help you?"></textarea>
      </div>
      <div id="contactFeedback"></div>
    `;
  }
  
  if (form) {
    form.addEventListener('submit', handleContactSubmit);
  }
}

async function handleReviewSubmit(event) {
  event.preventDefault();
  const form = event.target;
  const feedbackEl = document.getElementById('reviewFeedback');
  const submitBtn = form.querySelector('.review-submit-btn');
  
  const formData = {
    name: form.reviewName?.value?.trim(),
    rating: parseInt(form.reviewRating?.value) || 5,
    comment: form.reviewComment?.value?.trim()
  };
  
  if (!formData.name || !formData.comment) {
    if (feedbackEl) {
      feedbackEl.textContent = 'Please fill in your name and comment.';
      feedbackEl.className = 'promo-feedback promo-feedback-error';
    }
    return;
  }
  
  submitBtn.disabled = true;
  submitBtn.textContent = 'Submitting...';
  
  try {
    const response = await fetch(`${API_BASE}/api/reviews`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData)
    });
    
    const result = await response.json();
    
    if (result.success) {
      if (feedbackEl) {
        feedbackEl.textContent = result.message || 'Thank you for your review!';
        feedbackEl.className = 'promo-feedback promo-feedback-success';
      }
      form.reset();
      
      const reviews = await fetchApi('/api/reviews/latest');
      if (reviews) {
        renderReviews(reviews);
      }
    } else {
      throw new Error(result.error || 'Submission failed');
    }
  } catch (error) {
    console.error('Review submission error:', error);
    if (feedbackEl) {
      feedbackEl.textContent = 'Failed to submit review. Please try again.';
      feedbackEl.className = 'promo-feedback promo-feedback-error';
    }
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Submit Review';
  }
}

function initReviewForm() {
  const form = document.getElementById('reviewForm');
  if (form && !form.querySelector('input')) {
    form.innerHTML = `
      <div class="promo-form-group">
        <label for="reviewName">Your Name *</label>
        <input type="text" id="reviewName" name="reviewName" required placeholder="Your name">
      </div>
      <div class="promo-form-group">
        <label for="reviewRating">Rating</label>
        <select id="reviewRating" name="reviewRating">
          <option value="5">★★★★★ (5)</option>
          <option value="4">★★★★☆ (4)</option>
          <option value="3">★★★☆☆ (3)</option>
          <option value="2">★★☆☆☆ (2)</option>
          <option value="1">★☆☆☆☆ (1)</option>
        </select>
      </div>
      <div class="promo-form-group full">
        <label for="reviewComment">Your Review *</label>
        <textarea id="reviewComment" name="reviewComment" rows="3" required placeholder="Share your experience with us..."></textarea>
      </div>
      <div id="reviewFeedback"></div>
      <button type="submit" class="review-submit-btn">Submit Review</button>
    `;
  }
  
  if (form) {
    form.addEventListener('submit', handleReviewSubmit);
  }
}

async function initLandingPage() {
  initContactForm();
  
  const [hero, features, faqs, contact, story, reviews] = await Promise.all([
    fetchApi('/api/hero'),
    fetchApi('/api/features'),
    fetchApi('/api/faqs'),
    fetchApi('/api/contact'),
    fetchApi('/api/story'),
    fetchApi('/api/reviews/latest')
  ]);
  
  const content = {
    hero: hero || LANDING_CONTENT.hero,
    features: features || LANDING_CONTENT.features,
    faqs: faqs || LANDING_CONTENT.faqs,
    contact: contact || LANDING_CONTENT.contact,
    story: story || LANDING_CONTENT.story,
    reviews: reviews || LANDING_CONTENT.reviews
  };
  
  renderHero(content.hero);
  renderMenu(LANDING_CONTENT.menu);
  renderBusinessHours(LANDING_CONTENT.businessHours);
  renderFeatures(content.features);
  renderStory(content.story);
  renderReviews(content.reviews);
  renderFaqs(content.faqs);
  renderContact(content.contact);
  
  const ctaBtn = document.querySelector('.hero-cta');
  if (ctaBtn) {
    ctaBtn.addEventListener('click', handleCtaClick);
  }
}

document.addEventListener('DOMContentLoaded', initLandingPage);
