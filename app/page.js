import Createworkspace from "./pages/Createworkspace";
import Index from "./pages/Index";
// import Dashboard from './pages/admin/Dashboard';
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
export default function Home() {
  return (
    <main>
        {/* <Index /> */} 
        <Createworkspace/>  
        <ToastContainer />

        {/* <Dashboard /> */}
    </main>
  );
}
