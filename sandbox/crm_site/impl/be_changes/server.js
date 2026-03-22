import express from 'express';
import cors from 'cors';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(join(__dirname, '..', '..')));

const LANDING_CONTENT = {
  hero: {
    title: 'Elegance Fashion',
    subtitle: 'Discover timeless elegance and modern style. Premium fashion designed for the confident you.',
    ctaText: 'Explore Collection'
  },
  features: [
    {
      icon: 'quality',
      title: 'Premium Quality',
      description: 'Crafted with the finest materials and meticulous attention to detail for lasting elegance.'
    },
    {
      icon: 'design',
      title: 'Modern Design',
      description: 'Contemporary styles that blend classic sophistication with current trends.'
    },
    {
      icon: 'sustainability',
      title: 'Sustainable Fashion',
      description: 'Committed to ethical production and environmentally conscious practices.'
    }
  ],
  faqs: [
    {
      question: 'What is your return policy?',
      answer: 'We offer a 30-day return policy for all unworn items with original tags attached.'
    },
    {
      question: 'How do I find my correct size?',
      answer: 'Use our detailed size guide available on each product page. We also offer free virtual styling consultations.'
    },
    {
      question: 'Do you ship internationally?',
      answer: 'Yes, we ship to over 100 countries worldwide. Free shipping on orders over $200.'
    },
    {
      question: 'How do I care for my garments?',
      answer: 'Each item comes with specific care instructions. Most pieces are machine washable on gentle cycles.'
    }
  ],
  contact: {
    email: 'contact@elegancefashion.com',
    phone: '+86 400-888-8888',
    address: '888 Fashion Avenue, Style District, Shanghai, China'
  }
};

const BRAND_STORY = {
  title: 'Our Story',
  subtitle: 'A Legacy of Elegance',
  content: [
    {
      heading: 'Founded in 2010',
      paragraph: 'Elegance Fashion began with a simple vision: to create timeless pieces that empower women to feel confident and beautiful.'
    },
    {
      heading: 'Our Philosophy',
      paragraph: 'We believe fashion is more than clothing—it is self-expression. Each piece in our collection is designed to celebrate individuality and grace.'
    },
    {
      heading: 'Craftsmanship',
      paragraph: 'Every garment is crafted by skilled artisans using traditional techniques combined with modern innovation.'
    }
  ],
  timeline: [
    { year: '2010', event: 'Founded in Shanghai' },
    { year: '2015', event: 'Expanded to 50 stores worldwide' },
    { year: '2020', event: 'Launched sustainable fashion line' },
    { year: '2024', event: '100+ international locations' }
  ]
};

const REVIEWS = [
  {
    id: 1,
    name: 'Sarah Chen',
    rating: 5,
    comment: 'Absolutely love the quality! These pieces have become my wardrobe essentials.',
    date: '2024-01-15'
  },
  {
    id: 2,
    name: 'Emily Wang',
    rating: 5,
    comment: 'The design is elegant and sophisticated. Received many compliments!',
    date: '2024-01-10'
  },
  {
    id: 3,
    name: 'Jessica Liu',
    rating: 4,
    comment: 'Great customer service and beautiful packaging. Will definitely order again.',
    date: '2024-01-05'
  },
  {
    id: 4,
    name: 'Amanda Zhang',
    rating: 5,
    comment: 'The sustainable fashion line is amazing. Feel good about my purchase!',
    date: '2023-12-28'
  },
  {
    id: 5,
    name: 'Michelle Tan',
    rating: 5,
    comment: 'Perfect fit and excellent quality. Worth every penny!',
    date: '2023-12-20'
  }
];

const ICONS = {
  quality: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>',
  design: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>',
  sustainability: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22c4-4 8-7.5 8-12a8 8 0 1 0-16 0c0 4.5 4 8 8 12z"/><circle cx="12" cy="10" r="3"/></svg>'
};

app.get('/api/content', (req, res) => {
  res.json({
    success: true,
    data: LANDING_CONTENT
  });
});

app.get('/api/hero', (req, res) => {
  res.json({
    success: true,
    data: LANDING_CONTENT.hero
  });
});

app.get('/api/features', (req, res) => {
  res.json({
    success: true,
    data: LANDING_CONTENT.features
  });
});

app.get('/api/faqs', (req, res) => {
  res.json({
    success: true,
    data: LANDING_CONTENT.faqs
  });
});

app.get('/api/contact', (req, res) => {
  res.json({
    success: true,
    data: LANDING_CONTENT.contact
  });
});

app.post('/api/contact', (req, res) => {
  const { name, email, message } = req.body;
  
  if (!name || !email || !message) {
    return res.status(400).json({
      success: false,
      error: 'Missing required fields: name, email, message'
    });
  }
  
  console.log('Contact form submission:', { name, email, message });
  
  res.json({
    success: true,
    message: 'Thank you for your message! We will get back to you soon.'
  });
});

app.get('/api/icons', (req, res) => {
  res.json({
    success: true,
    data: ICONS
  });
});

app.get('/api/story', (req, res) => {
  res.json({
    success: true,
    data: BRAND_STORY
  });
});

app.get('/api/reviews', (req, res) => {
  res.json({
    success: true,
    data: REVIEWS
  });
});

app.get('/api/reviews/latest', (req, res) => {
  const latestReviews = REVIEWS.slice(0, 3);
  res.json({
    success: true,
    data: latestReviews
  });
});

app.post('/api/reviews', (req, res) => {
  const { name, rating, comment } = req.body;
  
  if (!name || !rating || !comment) {
    return res.status(400).json({
      success: false,
      error: 'Missing required fields: name, rating, comment'
    });
  }
  
  if (rating < 1 || rating > 5) {
    return res.status(400).json({
      success: false,
      error: 'Rating must be between 1 and 5'
    });
  }
  
  const newReview = {
    id: REVIEWS.length + 1,
    name,
    rating,
    comment,
    date: new Date().toISOString().split('T')[0]
  };
  
  console.log('New review:', newReview);
  
  res.json({
    success: true,
    message: 'Thank you for your review!',
    data: newReview
  });
});

app.get('*', (req, res) => {
  res.sendFile(join(__dirname, '..', '..', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});

export default app;
