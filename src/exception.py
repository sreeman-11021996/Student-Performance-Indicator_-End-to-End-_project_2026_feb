from src.logger import logging

def error_message_detail(error:Exception)-> str: 
    """
    Extracts detailed error message including file name and line number.
    """
    
    # _,_,exc_traceback = error_detail.exc_info() : error_detal is sys
    exc_traceback = error.__traceback__
    
    if exc_traceback is None:
        return f"Error occurred but no traceback available: {str(error)}"
    
    file_path = exc_traceback.tb_frame.f_code.co_filename
    line_number = exc_traceback.tb_lineno

    error_message = (f"Error occurred in python script name : [{file_path}] \n"
                     f"line number : [{line_number}] \n"
                     f"error message : [{str(error)}]\n")
                        
    return error_message


class CustomException(Exception):
    """
    Custom Exception class with detailed logging.

    Usage:
        try:
            ...
        except Exception as e:
            raise CustomException (e) from None # shows only custom error message
            raise CustomException (e) from e # shows both the original error and custom  
    """
    
    def __init__(self,error:Exception):
        super().__init__(error)
        self.error_message = error_message_detail(error=error)   
        
        # Log the error automatically
        logging.error(self.error_message, exc_info=True)
        
    def __str__(self):
        return self.error_message
