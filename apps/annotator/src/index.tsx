import ReactDOM from 'react-dom/client';
import Root from './components/root/root';
import reportWebVitals from './reportWebVitals';
// this was an old version // import 'font-awesome/css/font-awesome.min.css'; // npm install --save font-awesome

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <Root />
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
