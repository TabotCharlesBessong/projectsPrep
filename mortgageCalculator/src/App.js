import "./App.css";
import { FcHome } from "react-icons/fc";
import Form from "./components/Form";
import React from "react";

function App() {
	return (
		<div
			className="App container"
			style={{ maxWidth: 500, margin: "1rem auto" ,display: "flex", justifyContent: "center",flexDirection:'column',alignItems:'center'}}
		>
			<h1 className="display-1 my-5">
				<FcHome /> Mortgage Calculator{" "}
			</h1>
			<Form />
		</div>
	);
}

export default App;
