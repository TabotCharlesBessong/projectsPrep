import React from "react";
import ReactDOM from "react-dom";
import "./index.css";
import App from "./App";
import {ContextProvider} from './context/ContextProvider'
// import reportWebVitals from "./reportWebVitals";

ReactDOM.render(
	<React.StrictMode>
		<ContextProvider>
			<App />
		</ContextProvider>
	</React.StrictMode>,
	document.getElementById("root")
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// reportWebVitals();
