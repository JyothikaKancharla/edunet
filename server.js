import express from 'express';
import path from 'path';
import mongoose from 'mongoose';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const app = express();
const port = 3000;

// Get the directory name
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Middleware
app.use(express.static(__dirname)); // Serve static files
app.use(express.json()); // Parse JSON requests

// MongoDB Connection
mongoose.connect('mongodb://127.0.0.1:27017/students', {
    useNewUrlParser: true,
    useUnifiedTopology: true
}).then(() => {
    console.log("MongoDB connection successful");
}).catch((err) => {
    console.error("MongoDB connection error:", err);
});

// User Schema
const userSchema = new mongoose.Schema({
    firstName: { type: String, required: true },
    lastName: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true }
});

const User = mongoose.model('User', userSchema);

// **Register User**
app.post('/register', async (req, res) => {
    console.log("Incoming registration data:", req.body); // Debugging log
    try {
        const { firstName, lastName, email, password, confirmpassword } = req.body;

        if (!firstName || !lastName || !email || !password || !confirmpassword) {
            return res.status(400).json({ message: 'All fields are required' });
        }

        if (password !== confirmpassword) {
            return res.status(400).json({ message: 'Passwords do not match' });
        }

        const hashedPassword = await bcrypt.hash(password, 10);
        const newUser = new User({ firstName, lastName, email, password: hashedPassword });

        const savedUser = await newUser.save();
        console.log("User saved successfully:", savedUser); // Debugging log
        res.status(201).json({ message: 'User registered successfully' });
    } catch (error) {
        console.error("Error registering user:", error);
        res.status(500).json({ message: 'Error registering user', error });
    }
});

// **Login User**
app.post('/login', async (req, res) => {
    console.log("Incoming login data:", req.body); // Debugging log
    try {
        const { email, password, rememberMe } = req.body;
        const user = await User.findOne({ email });

        if (!user) return res.status(400).json({ message: 'User not found' });

        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) return res.status(400).json({ message: 'Invalid credentials' });

        const SECRET_KEY = 'your-secret-key'; // Replace with your actual secret key
        const token = jwt.sign({ id: user._id, email: user.email }, SECRET_KEY, {
            expiresIn: rememberMe ? '7d' : '1h'
        });

        res.status(200).json({ message: 'Login successful', token });
    } catch (error) {
        console.error("Error logging in:", error);
        res.status(500).json({ message: 'Error logging in', error });
    }
});

// Serve index.html


// Start Server
// app.listen(port, () => {
//     console.log(`Server started on http://localhost:${port}`);
// });

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'templates')));

// Serve index.html for the root route
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'templates/index.html'));
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
