
import { initializeApp, getApps, getApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getAuth, signInAnonymously } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyBWPQPsgO3Fnbt7x4zhdVh8IqnSmB46iAE",
  authDomain: "platego-fa6b8.firebaseapp.com",
  projectId: "platego-fa6b8",
  storageBucket: "platego-fa6b8.firebasestorage.app",
  messagingSenderId: "221736279772",
  appId: "1:221736279772:web:0dc3fa8fa32e0184291e19",
  measurementId: "G-JY05T8WVM4"
};

const app = getApps().length ? getApp() : initializeApp(firebaseConfig);

const auth = getAuth(app);

const db = getFirestore(app);

export { app, db, auth };
