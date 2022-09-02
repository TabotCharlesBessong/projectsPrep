import React,{useState,useEffect} from 'react'
import './App.css'
import { BrowserRouter,Routes,Route } from 'react-router-dom'
import {FiSettings} from 'react-icons/fi'
import { TooltipComponent } from '@syncfusion/ej2-react-popups'
import { Sidebar } from './components'

const App = () => {
  const [activeMenu,setActiveMenu] = useState(true)
  return (
		<div>
			<BrowserRouter>
				<div className="flex realtive dark:bg-main-dark-bg ">
					<div className="fixed right-4 bottom-4" style={{ zIndex: "1000" }}>
						<TooltipComponent content="Settings" position="Top">
							<button
								type="button"
								className="text-3xl p-3 hover:drop-shadow-xl hover:bg-light-gray text-white "
								style={{
									background: "blue",
									borderRadius: "50%",
								}}
							>
								<FiSettings />
							</button>
						</TooltipComponent>
					</div>
					{activeMenu ? (
						<div className="w-72 fixed dark:bg-secondary-dark-bg bg-white sidebar">
							sidebar
						</div>
					) : (
						<div className="w-0 dark:bg-secondary-dark-bg">hello</div>
					)}
					<div
						// className={
						// 	activeMenu
						// 		? "dark:bg-main-bg bg-main-bg min-h-screen md:ml-72 w-full "
						// 		: "dark:bg-main-bg bg-main-bg min-h-screen  w-full flex-2"
						// }
						className={` dark:bg-main-bg bg-main-bg min-h-screen  w-full ${
							activeMenu ? "md:ml-72" : "flex-2"
						} `}
					>
						<div className="fixed md:static bg-main-bg dark:bg-main-dark-bg navbar w-full ">
							Navbar
						</div>
					</div>
					<div>
						<Routes>
							{/* dashbaords */}
							<Route path="/" component={Sidebar} />
							<Route path="/ecommerce" component="employee" />

							{/* pages */}
							<Route path="/orders" component="Orders" />
							<Route path="/employees" component="Employees" />
							<Route path="/customers" component="Customers" />

              {/* apps */}
              <Route path='/kanban' component='Kanban' />
              <Route path='/editor' component='Editor' />
              <Route path='/charts' component='Charts' />
              <Route path='/calendar' component='Calendar' />
              <Route path='/cards' component='Cards' />
              <Route path='/color' component='Color' />

              {/* charts */}
              {/* Route for all types of suncfusion charts */}
              <Route path='/line' component='Line' />
              <Route path='/bar' component='Bar' />
              <Route path='/pie' component='Pie' />
              <Route path='/financial' component='Financial' />
              <Route path='/area' component='Area' />
              <Route path='/pyramid' component='Pyramid' />
              <Route path='/color-mapping' component='ColorMapping' />
              <Route path='/stack' component='Stack' />
						</Routes>
					</div>
				</div>
			</BrowserRouter>
			<h1 className="underline text-3xl">Hello world</h1>
		</div>
	);
}

export default App