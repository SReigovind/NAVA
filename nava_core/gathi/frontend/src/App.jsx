import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./components/AuthProvider.jsx";
import Layout from "./components/Layout.jsx";
import Landing from "./pages/Landing.jsx";
import Auth from "./pages/Auth.jsx";
import Fields from "./pages/Fields.jsx";
import FieldDetail from "./pages/FieldDetail.jsx";
import CropDetail from "./pages/CropDetail.jsx";

const RequireAuth = ({ children }) => {
  const { user, loading } = useAuth();
  if (loading) return <div className="page-center"><div className="spinner" /></div>;
  if (!user) return <Navigate to="/auth" replace />;
  return children;
};

// Crop detail needs full-height, no inner padding
const CropLayout = ({ children }) => (
  <Layout noPadding>{children}</Layout>
);

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/auth" element={<Auth />} />
        <Route path="/fields" element={<RequireAuth><Layout><Fields /></Layout></RequireAuth>} />
        <Route path="/fields/:fieldId" element={<RequireAuth><Layout><FieldDetail /></Layout></RequireAuth>} />
        <Route path="/fields/:fieldId/crops/:cropId" element={<RequireAuth><CropLayout><CropDetail /></CropLayout></RequireAuth>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AuthProvider>
  );
}

