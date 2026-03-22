const API_BASE = '';

const LANDING_CONTENT = {
  hero: {
    title: 'Artisan & Co.',
    subtitle: 'Timeless elegance crafted with passion. Discover our curated collection of sustainable fashion, where every piece tells a story of quality and style.',
    ctaText: 'Explore Collection'
  },
  features: [
    {
      icon: 'quality',
      title: 'Premium Quality',
      description: 'Each garment is crafted with the finest materials and meticulous attention to detail, ensuring lasting elegance.'
    },
    {
      icon: 'design',
      title: 'Thoughtful Design',
      description: 'Our designs blend contemporary trends with classic silhouettes, creating pieces that transcend seasons.'
    },
    {
      icon: 'sustainability',
      title: 'Sustainable Fashion',
      description: 'Committed to ethical practices and eco-friendly materials, for a wardrobe that cares for the planet.'
    }
  ],
  faqs: [
    {
      question: 'What makes Artisan & Co. different?',
      answer: 'We combine timeless design with sustainable practices, creating pieces that are both beautiful and responsible. Each item is crafted with premium materials and attention to detail.'
    },
    {
      question: 'What is your return policy?',
      answer: 'We offer a 30-day return policy for unworn items with tags attached. Simply contact our customer service to initiate a return.'
    },
    {
      question: 'Do you ship internationally?',
      answer: 'Yes, we ship to over 50 countries worldwide. Shipping times and costs vary by location.'
    },
    {
      question: 'How should I care for my garments?',
      answer: 'Each piece comes with specific care instructions. Generally, we recommend gentle washing and air drying to maintain quality.'
    }
  ],
  contact: {
    email: 'hello@artisanandco.com',
    phone: '+86 400-XXX-XXXX',
    address: '123 Fashion Avenue, Design District, Shanghai'
  },
  story: {
    title: 'Our Story',
    subtitle: 'A journey of craftsmanship and passion',
    content: [
      {
        heading: 'The Beginning',
        paragraph: 'Founded in 2010, Artisan & Co. began with a simple vision: to create clothing that celebrates the art of traditional craftsmanship while embracing modern aesthetics.'
      },
      {
        heading: 'Our Philosophy',
        paragraph: 'We believe that fashion should be sustainable, ethical, and beautiful. Every stitch tells a story of dedication and skill.'
      },
      {
        heading: 'Today',
        paragraph: 'Now a globally recognized brand, we continue to honor our roots while innovating for the future of sustainable fashion.'
      }
    ],
    timeline: [
      { year: '2010', event: 'Brand founded in Shanghai' },
      { year: '2015', event: 'First international store opens' },
      { year: '2018', event: 'Sustainability initiative launched' },
      { year: '2023', event: 'Awarded Best Ethical Brand' }
    ]
  },
  reviews: [
    { id: 1, name: 'Sarah M.', rating: 5, comment: 'Beautiful quality, the fabric feels amazing. Worth every penny.', date: '2024-01-15' },
    { id: 2, name: 'Emma L.', rating: 5, comment: 'Absolutely love the design. Gets compliments every time I wear it!', date: '2024-01-10' },
    { id: 3, name: 'Jessica K.', rating: 4, comment: 'Great fit and excellent customer service. Highly recommend!', date: '2024-01-05' }
  ]
};

const ICONS = {
  quality: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>',
  design: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 1 0 20"/><path d="M12 2a10 10 0 0 0 0 20"/></svg>',
  sustainability: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22c4-4 8-7.5 8-12a8 8 0 1 0-16 0c0 4.5 4 8 8 12z"/><circle cx="12" cy="10" r="3"/></svg>'
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
  
  initReviewForm();
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
    footerText.textContent = `Contact: ${contact.email} | ${contact.phone}`;
  }
}

function handleCtaClick() {
  const featuresSection = document.getElementById('features');
  if (featuresSection) {
    featuresSection.scrollIntoView({ behavior: 'smooth' });
  }
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
