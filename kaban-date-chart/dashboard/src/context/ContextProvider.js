
import React , {createContext,useContext,useState} from 'react'

const StateContext = createContext()

const initialState = {
  chat:false,
  cart:false,
  userProfile:false,
  notification:false
}

export const ContextProvider = ({children}) => {

  const [activeMenu,setActiveMenu] = useState(true)
  const [isClicked, setIsClicked] = useState(initialState);
  const [screenSize, setScreenSize] = useState(undefined);
	const [currentColor, setCurrentColor] = useState("#03C9D7");
	const [currentMode, setCurrentMode] = useState("Light");
	const [themeSettings, setThemeSettings] = useState(false);

  const setMode = (e) => {
		setCurrentMode(e.target.value);
		localStorage.setItem("themeMode", e.target.value);
	};

	const setColor = (color) => {
		setCurrentColor(color);
		localStorage.setItem("colorMode", color);
	};

  const handleClick = (clicked) =>
		setIsClicked({ ...initialState, [clicked]: true });

  return (
    <StateContext.Provider
    value={{
      activeMenu,
      setActiveMenu,
      currentColor,
      setCurrentColor,
      isClicked,
      setIsClicked,
      handleClick,
      setColor,
      setMode,
      screenSize,
      setScreenSize,
      currentMode,
      setCurrentMode,
      themeSettings,
      setThemeSettings
    }}
    >
      {children}
    </StateContext.Provider>
  )
}  

export const useStateContext = () => useContext(StateContext)