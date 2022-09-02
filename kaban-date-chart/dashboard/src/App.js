import React,{useState,useEffect} from 'react'
import './App.css'
import { BrowserRouter,Routes,Route } from 'react-router-dom'
import {FiSettings} from 'react-icons/fi'
import { TooltipComponent } from '@syncfusion/ej2-react-popups'

const App = () => {
  const [activeMenu,setActiveMenu] = useState(true)
  return (
    <div>
      <BrowserRouter>
        <div className='flex realtive dark:bg-main-dark-bg ' >
          <div className="fixed right-4 bottom-4" style={{zIndex:'1000'}} >
            <TooltipComponent content="Settings" position="Top"  >
              <button type='button' className='text-3xl p-3 hover:drop-shadow-xl hover:bg-light-gray text-white ' style={{
                background:'blue',
                borderRadius:'50%'
              }}  >
                <FiSettings/>
              </button>
            </TooltipComponent>
          </div>
          {
            activeMenu ? (
              <div>sidebar</div>
            ) : (
              <div>hello</div>
            )
          }
        </div>
      </BrowserRouter>
      <h1 className="underline text-3xl">Hello world</h1>
    </div>
  )
}

export default App