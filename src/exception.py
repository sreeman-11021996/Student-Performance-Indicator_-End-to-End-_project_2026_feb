import sys

def error_message_detail(error:Exception, error_detail:sys)-> str:
    
    _,_,exc_traceback = error_detail.exc_info()
    file_path = exc_traceback.tb_frame.f_code.co_filename
    line_number = exc_traceback.tb_lineno

    error_message = (f"Error occurred in python script name : [{file_path}] \n"
                     f"line number : [{line_number}] \n"
                     f"error message : [{str(error)}]\n")
                        
    return error_message


class CustomException(Exception):
    """
    use : raise CustomException(e,sys) from None : to avoid recursion of CustomException

    """
    def __init__(self,error, error_detail:sys):
        super().__init__(error)
        self.error_message = error_message_detail(error=error, error_detail=error_detail)   
        
        
    def __str__(self):
        return self.error_message
