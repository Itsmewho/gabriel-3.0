import { useNavigate } from 'react-router';
import styles from './styles/Notfound.module.css';

const NotFoundScreen = () => {
  const navigate = useNavigate();

  const GoBack = () => {
    navigate('/Dashboard');
  };

  return (
    <div className={styles.notfound_container}>
      <h1 className={styles.title}>404</h1>
      <h2 className={styles.subtitle}>Not Found</h2>
      <button className="btn fs-400" onClick={GoBack}>
        Go Back
      </button>
    </div>
  );
};

export default NotFoundScreen;
