import sys

def error_message_detail(error:Exception, error_detail:sys)-> str:
    _,_,exc_traceback = error_detail.exc_info()
    file_path = exc_traceback.tb_frame.f_code.co_filename
    line_no = exc_traceback.tb_lineno
    error_message = (
    "Error occurred in python script name [{0}] "
    "line number [{1}] "
    "error message [{2}]").format(file_path, line_no, str(error))
                        
    return error_message


class CustomException(Exception):
    def __init__(self,error, error_detail:sys):
        super().__init__(error)
        self.error_message = error_message_detail(error=error, error_detail=error_detail)
        
    def __str__(self):
        return self.error_message
    