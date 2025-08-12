import { Outlet } from 'react-router-dom';
import Footer from '@/components/Footer/Footer';
import './index.css';
import Header from './components/Header/Header';

function App() {
  return (
    <>
      <Header />
      <main>
        <Outlet></Outlet>
      </main>
      <Footer />
    </>
  );
}

export default App;
