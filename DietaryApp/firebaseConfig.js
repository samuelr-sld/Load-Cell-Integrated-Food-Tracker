
import { initializeApp, getApps, getApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { Platform } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { getAuth, initializeAuth, getReactNativePersistence } from "firebase/auth";

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

// initializeAuth throws if Auth already exists (Fast Refresh), so fall back to getAuth.
// Web has no getReactNativePersistence; its default persistence already survives restarts.
let auth;
try {
  auth = Platform.OS === "web"
    ? getAuth(app)
    : initializeAuth(app, { persistence: getReactNativePersistence(AsyncStorage) });
} catch (e) {
  auth = getAuth(app);
}

const db = getFirestore(app);

export { app, db, auth };
