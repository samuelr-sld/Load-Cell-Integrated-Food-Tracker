import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = {
  apiKey: "AIzaSyCUE6HB-4JdFCCV1Gvk4zkajW61K4RgbiM",
  authDomain: "electives-mobile-application.firebaseapp.com",
  projectId: "electives-mobile-application",
  storageBucket: "electives-mobile-application.firebasestorage.app",
  messagingSenderId: "806595189071",
  appId: "1:806595189071:web:0a4c90b21177228c6bcd3a",
  measurementId: "G-QTF5TCS3V5"
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);